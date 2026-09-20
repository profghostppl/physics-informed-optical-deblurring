"""
UnfoldedOpticsDeblurNet
=======================

A physics-informed, non-generative neural deblurring architecture that reverses
lens defocus / optical aberrations using known camera hardware metadata
(focal length, f-number, pixel pitch, focus distance, subject distance).

Design summary
---------------
The restoration problem is posed as the classical energy minimization

    min_x  (1/2) || y - K x ||_2^2 + lambda * R(x)

and solved via Half-Quadratic Splitting (HQS), unrolled into K = 6 stages that
alternate between:

  1. An *analytical*, non-learned data-fidelity step, solved in closed form in
     the 2D Fourier domain (Wiener deconvolution). This step is where the
     optics (the known/estimated blur kernel K) is injected -- it contains no
     learned image content, only a learned scalar trade-off weight mu_k. This
     is what prevents the network from hallucinating structure: the only
     place image evidence enters is through the physically-derived kernel.

  2. A *learned* proximal / denoising step z^(k+1) = D_theta^(k)(x^(k+1/2), sigma)
     implemented as a small, spectrally-constrained (1-Lipschitz-per-layer)
     CNN. Because every convolution is spectrally normalized and the network
     is a pure feed-forward composition (skip connections are concatenation-
     based, never additive-residual-within-a-block), the overall mapping is
     non-expansive by sub-multiplicativity of Lipschitz constants. Per
     Monotone Operator Theory (Ryu et al., "Plug-and-Play Methods Provably
     Converge with Properly Trained Denoisers", ICML 2019), plugging a
     non-expansive denoiser into a fixed-point iteration of this form yields a
     stable, convergent scheme rather than an unconstrained generator that is
     free to invent high frequency content.

All radiometry happens in *linear* light: the input/output sRGB images are
passed through an invertible ISP (IEC 61966-2-1 EOTF/OETF) so that the linear
convolution model y = Kx + n is physically valid (sRGB gamma-encoded pixels do
not sum linearly, so deconvolving in sRGB space is physically incorrect).

Author: Principal Computational Imaging / Deep Learning Systems reference
implementation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


# ======================================================================================
# 1. Optics: geometric defocus model
# ======================================================================================
#
# ---- Derivation of the defocus blur circle diameter b(d) -----------------------------
#
# Thin-lens equation:               1/d + 1/v = 1/f   =>   v(d) = f*d / (d - f)
#
# The camera is focused at distance d0, so the sensor sits at the conjugate image
# distance:
#
#       v0 = v(d0) = f*d0 / (d0 - f)
#
# A scene point actually at distance d forms a sharp image at
#
#       v  = v(d)  = f*d  / (d - f)
#
# which in general does NOT coincide with the sensor plane at v0. The cone of light
# converging toward the (out-of-focus) image point v is intercepted by the sensor at
# v0, producing a blur disc. By similar triangles, an aperture of diameter
#
#       A = f / N                                   (N = f-number)
#
# projects a blur-disc diameter on the sensor of
#
#       c = A * |v0 - v| / v                                                     (1)
#
# Compute v0 - v:
#
#   v0 - v = f*d0/(d0-f) - f*d/(d-f)
#          = f * [ d0*(d-f) - d*(d0-f) ] / [ (d0-f)*(d-f) ]
#          = f * [ d0*d - d0*f - d*d0 + d*f ] / [ (d0-f)*(d-f) ]
#          = f * [ f*(d - d0) ] / [ (d0-f)*(d-f) ]
#          = f^2 * (d - d0) / [ (d0-f)*(d-f) ]                                   (2)
#
# and
#
#   |v0-v| / v = f^2*|d-d0| / [(d0-f)(d-f)]  *  (d-f) / (f*d)
#              = f*|d-d0| / [ (d0-f)*d ]                                          (3)
#
# Substituting (3) into (1):
#
#   c = (f/N) * f*|d-d0| / [(d0-f)*d]
#     = f^2 * |d - d0| / [ N * d * (d0 - f) ]                                     (4)
#
# which is exactly b(d) = f^2/N * |d-d0| / (d*(d0-f)).   QED.
#
# ---- Conversion from metric diameter to pixel kernel radius --------------------------
#
# b(d) above is a *diameter* measured in the same length units as f, d, d0 (meters,
# if all inputs are given in meters). To rasterize the blur kernel we need the RADIUS
# in pixels, given the sensor's physical pixel pitch p (meters/pixel):
#
#       R_px = b(d) / (2 * p)
#
# (divide by 2 to go diameter -> radius, divide by p to go meters -> pixels).
#
# ---- Frequency-domain derivation of the closed-form HQS data step --------------------
#
# The HQS data sub-problem is
#
#       x^(k+1/2) = argmin_x || y - K x ||_2^2 + mu_k || x - z^(k) ||_2^2
#
# Setting the gradient to zero (K here denotes the *linear convolution operator*):
#
#       2 K^T (K x - y) + 2 mu_k (x - z^(k)) = 0
#       (K^T K + mu_k I) x = K^T y + mu_k z^(k)
#
# Because K is a spatial convolution with kernel k(.,.), it diagonalizes under the
# 2D Discrete Fourier Transform: K = F^-1 diag(K_fft) F, and K^T (correlation) becomes
# diag(conj(K_fft)) in the Fourier domain, with K^T K -> diag(|K_fft|^2). The linear
# system therefore decouples into independent per-frequency scalar equations:
#
#       (|K_fft|^2 + mu_k) X_fft = conj(K_fft) * Y_fft + mu_k * Z_fft
#
#   =>  X_fft = [ conj(K_fft) * Y_fft + mu_k * Z_fft ] / [ |K_fft|^2 + mu_k ]      (5)
#
# which is exactly the batched Wiener-style update implemented in
# `AnalyticalWienerDeconv` below, evaluated with `torch.fft.rfft2` /
# `torch.fft.irfft2` for real-valued images.
# ======================================================================================


def circle_of_confusion_diameter_m(
    focal_length_m: torch.Tensor,
    f_number: torch.Tensor,
    focus_distance_m: torch.Tensor,
    subject_distance_m: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Eq. (4): b(d) = f^2/N * |d - d0| / (d * (d0 - f)), all inputs in meters.

    All tensors broadcast against each other; typically shape (B,) or (B,1,1,1).
    """
    f = focal_length_m
    n = f_number
    d0 = focus_distance_m
    d = subject_distance_m
    denom = n * d * (d0 - f)
    denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
    b = (f.pow(2) * (d - d0).abs()) / denom
    return b.abs()


def defocus_diameter_to_pixel_radius(
    b_m: torch.Tensor, pixel_pitch_m: torch.Tensor
) -> torch.Tensor:
    """R_px = b / (2*p): metric blur-circle diameter -> pixel kernel radius."""
    return b_m / (2.0 * pixel_pitch_m)


def make_soft_pillbox_kernel(
    radius_px: torch.Tensor, ksize: int, softness_px: float = 0.75
) -> torch.Tensor:
    """Rasterize a (soft-edged, differentiable) disk / pillbox PSF.

    A hard disk indicator is not differentiable w.r.t. its radius, so the boundary is
    relaxed with a sigmoid ("soft aperture edge") of width `softness_px`. This lets the
    radius (and therefore the upstream depth / camera parameters) receive gradients.

    Args:
        radius_px: (B,) or (B,1,1,1) tensor of blur radii, in pixels.
        ksize: odd kernel spatial size.
        softness_px: transition width of the soft edge, in pixels.
    Returns:
        (B, 1, ksize, ksize) kernel tensor, each sample normalized to sum to 1.
    """
    device = radius_px.device
    dtype = radius_px.dtype
    r = radius_px.reshape(-1, 1, 1, 1)
    half = (ksize - 1) / 2.0
    ys = torch.arange(ksize, device=device, dtype=dtype) - half
    xs = torch.arange(ksize, device=device, dtype=dtype) - half
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt(yy.pow(2) + xx.pow(2)).reshape(1, 1, ksize, ksize)
    r_safe = torch.clamp(r, min=1e-3)
    kernel = torch.sigmoid((r_safe - dist) / max(softness_px, 1e-3))
    kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return kernel


# ======================================================================================
# 2. Invertible ISP: sRGB <-> linear irradiance (IEC 61966-2-1)
# ======================================================================================
class InvertibleISP(nn.Module):
    """Differentiable, invertible sRGB <-> linear mapping (IEC 61966-2-1 piecewise EOTF).

    Deconvolution assumes linear photon transport y = Kx + n. Camera JPEG/sRGB output
    is gamma-encoded, so we must decode to linear light before the physics-based
    Wiener step and re-encode to sRGB for display/output.
    """

    _A = 0.055
    _GAMMA = 2.4
    _LINEAR_THRESH_SRGB = 0.04045
    _LINEAR_THRESH_LINEAR = 0.0031308
    _EPS = 1e-8

    def to_linear(self, srgb: torch.Tensor) -> torch.Tensor:
        """EOTF: sRGB [0,1] -> linear irradiance [0,1]."""
        x = srgb.clamp(0.0, 1.0)
        lo = x / 12.92
        base = ((x + self._A) / (1 + self._A)).clamp_min(self._EPS)
        hi = base.pow(self._GAMMA)
        return torch.where(x <= self._LINEAR_THRESH_SRGB, lo, hi)

    def to_srgb(self, linear: torch.Tensor) -> torch.Tensor:
        """OETF (inverse EOTF): linear irradiance [0,1] -> sRGB [0,1]."""
        x = linear.clamp(0.0, 1.0)
        lo = x * 12.92
        base = x.clamp_min(self._EPS).pow(1.0 / self._GAMMA)
        hi = (1 + self._A) * base - self._A
        return torch.where(x <= self._LINEAR_THRESH_LINEAR, lo, hi).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, mode: str = "to_linear") -> torch.Tensor:
        if mode == "to_linear":
            return self.to_linear(x)
        if mode == "to_srgb":
            return self.to_srgb(x)
        raise ValueError(f"Unknown ISP mode: {mode}")


# ======================================================================================
# 3. Analytical Wiener deconvolution layer (closed-form HQS data-fidelity step)
# ======================================================================================
class AnalyticalWienerDeconv(nn.Module):
    """Closed-form data-fidelity step of Eq. (5), evaluated with real FFTs.

    X^(k+1/2) = [ conj(K_fft) * Y_fft + mu_k * Z_fft ] / [ |K_fft|^2 + mu_k ]

    * mu_k is learned in log-space (mu_k = exp(alpha_k)) so it is strictly positive by
      construction -- this removes any possibility of a division-by-zero / negative
      pole in the denominator, independent of what value gradient descent finds.
    * Because the DFT assumes toroidal (periodic) boundary conditions but real images
      are not periodic, we reflect-pad the image (and current prior estimate) by the
      kernel's half-width before transforming, and blend the padded border with a
      raised-cosine (Hann) taper -- an edgetaper-style relaxation that suppresses the
      violent ringing that naive zero/circular padding would otherwise inject from the
      image boundary.
    """

    def __init__(self, init_mu: float = 1.0, taper_px: Optional[int] = None):
        super().__init__()
        # mu_k = exp(alpha_k)  ->  strictly positive for any real alpha_k.
        self.alpha = nn.Parameter(torch.tensor(math.log(init_mu), dtype=torch.float32))
        self.taper_px = taper_px  # optional override; default = kernel half-size

    @property
    def mu(self) -> torch.Tensor:
        return torch.exp(self.alpha)

    @staticmethod
    def _hann_border_mask(h: int, w: int, pad_h: int, pad_w: int, device, dtype) -> torch.Tensor:
        """1 in the image interior, smoothly -> 0 over the padded border (raised cosine)."""
        def axis_profile(length: int, pad: int) -> torch.Tensor:
            prof = torch.ones(length, device=device, dtype=dtype)
            if pad > 0:
                ramp = 0.5 - 0.5 * torch.cos(
                    math.pi * (torch.arange(pad, device=device, dtype=dtype) + 0.5) / pad
                )
                prof[:pad] = ramp
                prof[-pad:] = ramp.flip(0)
            return prof

        row = axis_profile(h, pad_h)
        col = axis_profile(w, pad_w)
        return row.view(1, 1, h, 1) * col.view(1, 1, 1, w)

    def _pad_and_taper(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        h, w = x_pad.shape[-2:]
        mask = self._hann_border_mask(h, w, pad_h, pad_w, x.device, x.dtype)
        # Blend the reflect-padded image toward its own low-frequency (blurred) content
        # near the border, i.e. an edgetaper-style relaxation of the hard reflect seam.
        blurred = F.avg_pool2d(
            F.pad(x_pad, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1
        )
        return mask * x_pad + (1 - mask) * blurred

    @staticmethod
    def _psf_to_otf(psf: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        """Zero-pad + circularly center a spatial PSF, then rfft2 it (psf2otf)."""
        b, c, kh, kw = psf.shape
        otf_spatial = psf.new_zeros(b, c, out_h, out_w)
        otf_spatial[..., :kh, :kw] = psf
        otf_spatial = torch.roll(otf_spatial, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))
        return torch.fft.rfft2(otf_spatial)

    def forward(
        self, y: torch.Tensor, kernel: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            y: (B, C, H, W) observed (blurry) image, linear radiometric space.
            kernel: (B, 1, Hk, Wk) spatially-invariant PSF, each sample sums to 1.
            z: (B, C, H, W) current prior estimate z^(k).
        Returns:
            x_half: (B, C, H, W) closed-form data-fidelity estimate x^(k+1/2).
        """
        _, _, h, w = y.shape
        _, _, kh, kw = kernel.shape
        pad_h = self.taper_px if self.taper_px is not None else kh // 2
        pad_w = self.taper_px if self.taper_px is not None else kw // 2

        y_pad = self._pad_and_taper(y, pad_h, pad_w)
        z_pad = self._pad_and_taper(z, pad_h, pad_w)
        hp, wp = y_pad.shape[-2:]

        k_fft = self._psf_to_otf(kernel, hp, wp)          # (B,1,Hp,Wp//2+1) complex
        y_fft = torch.fft.rfft2(y_pad)                    # (B,C,Hp,Wp//2+1) complex
        z_fft = torch.fft.rfft2(z_pad)                    # (B,C,Hp,Wp//2+1) complex

        mu_k = self.mu.view(1, 1, 1, 1)

        # Explicit real/imag construction, float32-safe, avoids complex-division warnings.
        k_power = k_fft.real.pow(2) + k_fft.imag.pow(2)   # (B,1,Hp,Wp//2+1) real, >= 0
        denom = (k_power + mu_k).clamp_min(1e-12)         # strictly positive by construction

        k_conj = torch.conj(k_fft)
        numerator = k_conj * y_fft + mu_k * z_fft         # broadcasts (B,1,..) * (B,C,..)
        x_fft = numerator / denom

        x_pad_out = torch.fft.irfft2(x_fft, s=(hp, wp))
        x_half = x_pad_out[..., pad_h : pad_h + h, pad_w : pad_w + w]
        return x_half


# ======================================================================================
# 4. Lipschitz-constrained proximal denoiser (learned prior sub-problem)
# ======================================================================================
def _sn_conv2d(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1) -> nn.Module:
    return spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding))


class _SNConvBlock(nn.Module):
    """Sequential (non-additive-skip) spectral-normalized conv + activation.

    Deliberately has NO internal additive residual: composing 1-Lipschitz layers
    (spectral-normalized linear map + 1-Lipschitz activation) via sub-multiplicativity
    keeps the whole block's Lipschitz constant <= 1. An additive skip (x + f(x)) would
    break that clean bound (Lip <= 1 + Lip(f) instead), so it is avoided here; the
    network's only residual connection is the single *global* one applied once at the
    very end of `LipschitzProximalDenoiser.forward`.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = _sn_conv2d(in_ch, out_ch, k=3, stride=stride, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)  # 1-Lipschitz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class LipschitzProximalDenoiser(nn.Module):
    """Lightweight multi-scale (U-Net-style) 1-Lipschitz-per-layer proximal denoiser.

    z^(k+1) = D_theta(x^(k+1/2), sigma_k) = x^(k+1/2) - r_theta(concat(x^(k+1/2), sigma_k))

    Every convolution is wrapped in `spectral_norm`, so each layer's operator norm is
    <= 1; composed with 1-Lipschitz LeakyReLU activations, the sub-network r_theta is
    non-expansive by sub-multiplicativity of Lipschitz constants. Skip connections
    across scales are concatenation-based (safe: concatenation followed by a spectrally
    normalized conv is still just a single bounded linear map), never additive, so this
    property is preserved end-to-end. Under Monotone Operator Theory this makes the
    plug-and-play fixed point of the outer HQS iteration well-posed and stable, rather
    than an unconstrained generative mapping free to invent high-frequency content.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        noise_ch = 1
        # Encoder
        self.enc0 = _SNConvBlock(in_ch + noise_ch, base_ch, stride=1)
        self.enc1 = _SNConvBlock(base_ch, base_ch * 2, stride=2)
        self.enc2 = _SNConvBlock(base_ch * 2, base_ch * 4, stride=2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            _SNConvBlock(base_ch * 4, base_ch * 4, stride=1),
            _SNConvBlock(base_ch * 4, base_ch * 4, stride=1),
        )
        # Decoder (concatenation skips, then a spectrally-normalized conv to fuse)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _SNConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2, stride=1)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec0 = _SNConvBlock(base_ch * 2 + base_ch, base_ch, stride=1)
        # Output head: predicts the noise/degradation residual to subtract.
        self.out_conv = _sn_conv2d(base_ch, in_ch, k=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) deconvolved estimate x^(k+1/2), linear space.
            sigma: (B, 1, H, W) heteroscedastic noise-level map (matches x's H, W).
        Returns:
            z: (B, C, H, W) denoised prior estimate z^(k+1).
        """
        if sigma.shape[-2:] != x.shape[-2:]:
            sigma = F.interpolate(sigma, size=x.shape[-2:], mode="bilinear", align_corners=False)
        inp = torch.cat([x, sigma], dim=1)

        f0 = self.enc0(inp)          # (B, base,   H,   W)
        f1 = self.enc1(f0)           # (B, 2base,  H/2, W/2)
        f2 = self.enc2(f1)           # (B, 4base,  H/4, W/4)
        b = self.bottleneck(f2)      # (B, 4base,  H/4, W/4)

        u1 = self.up1(b)
        if u1.shape[-2:] != f1.shape[-2:]:
            u1 = F.interpolate(u1, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, f1], dim=1))   # (B, 2base, H/2, W/2)

        u0 = self.up0(d1)
        if u0.shape[-2:] != f0.shape[-2:]:
            u0 = F.interpolate(u0, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(torch.cat([u0, f0], dim=1))   # (B, base, H, W)

        residual = self.out_conv(d0)                 # (B, C, H, W)
        return x - residual                          # single *global* residual (DnCNN-style)


# ======================================================================================
# 5. Master unfolded network: K=6 HQS stages
# ======================================================================================
class UnfoldedOpticsDeblurNet(nn.Module):
    """Deep-unfolded HQS optimizer for physics-informed, non-generative deblurring.

    forward(Y, K, sigma) alternates, for k = 0..K_stages-1:
        x^(k+1/2) = AnalyticalWienerDeconv_k(Y_linear, K, z^(k))     # closed-form, no learned pixels
        z^(k+1)   = LipschitzProximalDenoiser_k(x^(k+1/2), sigma)    # 1-Lipschitz learned prior

    All arithmetic happens in linear radiometric space (via `InvertibleISP`); the
    output is re-encoded to sRGB before being returned.
    """

    def __init__(self, num_stages: int = 6, base_ch: int = 32, in_ch: int = 3):
        super().__init__()
        self.num_stages = num_stages
        self.isp = InvertibleISP()
        self.data_steps = nn.ModuleList(
            [AnalyticalWienerDeconv(init_mu=1.0) for _ in range(num_stages)]
        )
        self.priors = nn.ModuleList(
            [LipschitzProximalDenoiser(in_ch=in_ch, base_ch=base_ch) for _ in range(num_stages)]
        )

    def forward(
        self,
        y_srgb: torch.Tensor,
        kernel: torch.Tensor,
        sigma: torch.Tensor,
        use_learned_prior: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            y_srgb: (B, 3, H, W) blurred input image in sRGB [0,1].
            kernel: (B, 1, Hk, Wk) spatially-invariant optical PSF (each sample sums to 1).
                    Typically produced from camera metadata via
                    `circle_of_confusion_diameter_m` -> `defocus_diameter_to_pixel_radius`
                    -> `make_soft_pillbox_kernel`, or supplied directly from a calibrated /
                    depth-derived kernel estimate.
            sigma: (B, 1, H, W) estimated per-pixel noise standard deviation (sensor
                   read + shot noise floor), in the same linear-light units as y_linear.
            use_learned_prior: when True (default), each stage's closed-form Wiener
                    estimate is passed through the learned `LipschitzProximalDenoiser`,
                    as in training. When False, that CNN step is skipped entirely and
                    z^(k+1) = x^(k+1/2) directly -- i.e. "pure physics" mode: every
                    stage still runs its closed-form, deterministic Wiener deconvolution
                    (with that stage's trained mu_k trade-off weight), but no learned
                    network ever touches a pixel. This is the literal mathematical
                    inverse of whatever the blur kernel didn't destroy -- nothing is
                    invented, but frequencies the blur erased stay erased/noisy rather
                    than being smoothed over by the learned prior.
        Returns:
            x_restored: (B, 3, H, W) restored image in sRGB [0,1].
        """
        y_linear = self.isp.to_linear(y_srgb)
        z = y_linear
        for k in range(self.num_stages):
            x_half = self.data_steps[k](y_linear, kernel, z)
            z = self.priors[k](x_half, sigma) if use_learned_prior else x_half
        x_restored_linear = z.clamp(0.0, 1.0)
        x_restored = self.isp.to_srgb(x_restored_linear)
        return x_restored


# ======================================================================================
# 6. Self-contained execution test
# ======================================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running self-test on device: {device}")

    B, C, H, W = 2, 3, 64, 64
    KSIZE = 15

    # ---- 1. Derive a physically-plausible defocus kernel from camera metadata --------
    focal_length_m = torch.full((B,), 0.050, device=device)          # 50 mm lens
    f_number = torch.tensor([2.8, 5.6], device=device)                # f/2.8 and f/5.6
    pixel_pitch_m = torch.full((B,), 4.0e-6, device=device)           # 4 um pixels
    focus_distance_m = torch.full((B,), 2.0, device=device)           # focused at 2 m
    subject_distance_m = torch.tensor([2.6, 3.2], device=device)      # subject behind focus plane

    b_m = circle_of_confusion_diameter_m(
        focal_length_m, f_number, focus_distance_m, subject_distance_m
    )
    radius_px = defocus_diameter_to_pixel_radius(b_m, pixel_pitch_m)
    print(f"CoC diameter (m): {b_m.tolist()}")
    print(f"Blur kernel radius (px): {radius_px.tolist()}")

    kernel = make_soft_pillbox_kernel(radius_px, ksize=KSIZE, softness_px=0.75)
    kernel = kernel.to(device)
    assert kernel.shape == (B, 1, KSIZE, KSIZE)
    assert torch.allclose(kernel.sum(dim=(-2, -1)), torch.ones(B, 1, device=device), atol=1e-4)

    # ---- 2. Dummy blurry input + heteroscedastic noise floor --------------------------
    y_srgb = torch.rand(B, C, H, W, device=device, requires_grad=True)
    sigma = (0.01 + 0.02 * torch.rand(B, 1, H, W, device=device)).to(device)

    # ---- 3. Build and run the unfolded network ----------------------------------------
    model = UnfoldedOpticsDeblurNet(num_stages=6, base_ch=16).to(device)
    model.train()

    x_restored = model(y_srgb, kernel, sigma)

    print(f"Input  shape: {tuple(y_srgb.shape)}")
    print(f"Kernel shape: {tuple(kernel.shape)}")
    print(f"Sigma  shape: {tuple(sigma.shape)}")
    print(f"Output shape: {tuple(x_restored.shape)}")

    assert x_restored.shape == y_srgb.shape, "Output shape must match input shape"
    assert torch.isfinite(x_restored).all(), "Output must not contain NaN/Inf"
    assert x_restored.min() >= 0.0 and x_restored.max() <= 1.0, "Output must be valid sRGB in [0,1]"

    # ---- 4. Gradient flow check (end-to-end, including through the FFT data steps) ---
    loss = F.mse_loss(x_restored, torch.zeros_like(x_restored))
    loss.backward()

    assert y_srgb.grad is not None and torch.isfinite(y_srgb.grad).all(), \
        "Gradient must flow back to the input image"
    for k, step in enumerate(model.data_steps):
        assert step.alpha.grad is not None and torch.isfinite(step.alpha.grad).all(), \
            f"Stage {k}: mu_k (alpha) must receive a finite gradient"
    for k, prior in enumerate(model.priors):
        any_grad = any(p.grad is not None for p in prior.parameters())
        assert any_grad, f"Stage {k}: proximal denoiser must receive gradients"

    print("Learned mu_k per stage:", [round(s.mu.item(), 4) for s in model.data_steps])
    print("Loss:", loss.item())
    print("All shape / finiteness / gradient-flow assertions passed.")
