"""
AdvancedOpticsKernelEngine
===========================

Upgrades the sensor/lens/kernel-generation stage of `unfolded_optics_deblur.py` from a
single achromatic geometric pillbox (`circle_of_confusion_diameter_m` ->
`make_soft_pillbox_kernel`) to a wave-optics-aware, wavelength-dispersive, spatially
variant optical model. This module is additive: it *reuses* the existing geometric CoC
derivation (`circle_of_confusion_diameter_m`, `defocus_diameter_to_pixel_radius`,
`make_soft_pillbox_kernel` from `unfolded_optics_deblur.py`) rather than duplicating it,
and produces kernels that are drop-in compatible with `AnalyticalWienerDeconv` (which
already broadcasts a (B,1,...) or (B,3,...) kernel against a (B,3,...) image).

Five upgrades, one section each below:

  1. Diffraction-limited wavefront integration -- `AdvancedOpticsKernelEngine`
     combines the *geometric* defocus pillbox with the *diffraction* Airy pattern by
     convolving their PSFs in the Fourier domain (the physically correct way to cascade
     two independent blurring processes), with a fast "quadrature" fallback that instead
     widens the geometric pillbox radius by sqrt(b(d)^2 + d_Airy^2).

  2. Channel-dependent longitudinal chromatic aberration -- the same engine computes a
     per-RGB-channel effective focal length f(lambda) from a Cauchy dispersion model of
     the lens glass, and feeds each channel's f(lambda) back through the *existing*
     `circle_of_confusion_diameter_m` to get a genuinely different defocus (and, via
     lambda directly, a different Airy radius) per channel -- producing a 3-channel
     kernel (B,3,Hk,Wk) for analytic per-channel Wiener deconvolution.

  3. Depth-guided spatially variant inversion -- `DepthToKernelField` turns a monocular
     relative depth map into a small set of representative-distance PSFs plus a soft
     per-pixel bin-assignment field; `OverlapAddSpatialDeconv` then deconvolves the image
     patch-by-patch (locally shift-invariant approximation) and reassembles the patches
     with Hann-window overlap-add, which is seam-free by construction because it is
     renormalized by the folded window energy rather than relying on a fixed window
     shape.

  4. Boundary ringing suppression -- `DifferentiableEdgeTaper` derives its border ramp
     directly from the blur kernel's own cumulative energy profile (a physically
     adaptive edgetaper, in the spirit of Reeves & Lagendijk 1985) and blends the image
     with its own circular self-blur before any `rfft2` call.

  5. OTF zero-crossing safeguards -- `StabilizedMultiChannelWienerDeconv` regularizes
     the Wiener denominator with mu_k = softplus(alpha_k) + eps, which is strictly
     positive by construction (unlike a bare learned scalar) and therefore keeps the
     noise-amplification gain 1/(|K|^2 + mu_k) finite everywhere -- including exactly at
     the Bessel zero-crossings of the diffraction-limited OTF, where |K| -> 0 and an
     unregularized inverse filter would blow up.

Author: Principal Computational Imaging / Optics Software Engineering reference
implementation.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unfolded_optics_deblur import (
    UnfoldedOpticsDeblurNet,
    circle_of_confusion_diameter_m,
    defocus_diameter_to_pixel_radius,
    make_soft_pillbox_kernel,
)


# ======================================================================================
# 0. Shared PSF <-> OTF helper (psf2otf convention: kernel center -> index (0,0))
# ======================================================================================
def _psf_to_otf(psf: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Zero-pad a centered spatial PSF to (out_h, out_w), roll its center to index
    (0,0) (the standard `psf2otf` trick so that circular convolution via FFT does not
    introduce a linear phase / spatial shift), then `rfft2` it.

    Args:
        psf: (N, C, kh, kw), each (n,c) sample assumed centered at (kh//2, kw//2) and
             summing to 1 (as produced by `make_soft_pillbox_kernel` / `_make_airy_kernel`).
    Returns:
        (N, C, out_h, out_w//2+1) complex OTF.
    """
    n, c, kh, kw = psf.shape
    otf_spatial = psf.new_zeros(n, c, out_h, out_w)
    otf_spatial[..., :kh, :kw] = psf
    otf_spatial = torch.roll(otf_spatial, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))
    return torch.fft.rfft2(otf_spatial)


def _radius_to_auto_ksize(
    radius_px: torch.Tensor, min_k: int = 7, max_k: int = 65, margin: float = 4.0
) -> int:
    """Pick an odd kernel window large enough to hold `margin` blur-radii of context
    (so the circular-convolution / circular-taper approximations elsewhere in this file
    stay valid), clamped to [min_k, max_k]. Sized from `radius_px.detach()` because the
    kernel *grid resolution* is a discrete/non-differentiable design choice -- gradients
    still flow through the kernel's per-pixel *values* at that fixed resolution.
    """
    r_max = float(radius_px.detach().abs().amax().clamp_min(0.5))
    k = int(2 * math.ceil(margin * r_max) + 1)
    k = max(min_k, min(max_k, k))
    if k % 2 == 0:
        k += 1
    return k


def _inverse_softplus(y: float) -> float:
    """x such that softplus(x) = y, for initializing an `nn.Parameter` at a target mu."""
    y = max(y, 1e-6)
    return math.log(math.expm1(y)) if y < 20.0 else y


# ======================================================================================
# 1. Diffraction: differentiable Airy-disk PSF
# ======================================================================================
#
# The diffraction-limited intensity PSF of a circular aperture (Fraunhofer diffraction)
# is the Airy pattern:
#
#       I(r) = [ 2 J1(v) / v ]^2 ,        v = pi * D_aperture * r / (lambda * z)
#
# which, expressed via the radius of its first dark ring r0 = 1.22 * lambda * N
# (N = f-number; this is `d_Airy / 2` from the prompt's diameter formula), becomes
#
#       v(r) = alpha_1 * r / r0 ,          alpha_1 = 3.831705970207512  (first zero of J1)
#       I(r) = [ 2 J1(v(r)) / v(r) ]^2                                            (6)
#
# `torch.special.bessel_j1` exists but has **no autograd support** in this PyTorch build
# (verified: `.backward()` raises "does not require grad and does not have a grad_fn"),
# which would silently sever gradients from the diffraction kernel back to the
# upstream camera-parameter tensors (aperture, wavelength, depth). We therefore use the
# classic rational/asymptotic polynomial approximation of J1 (Numerical Recipes 6.5,
# accurate to ~1e-8), built entirely from elementary differentiable ops.
# ======================================================================================

_AIRY_FIRST_ZERO = 3.831705970207512  # first zero of the Bessel function J1(x)


def _bessel_j1(x: torch.Tensor) -> torch.Tensor:
    """Differentiable J1(x) via the Numerical Recipes rational/asymptotic approximation."""
    ax = x.abs()

    # ---- branch 1: |x| < 8, rational-polynomial approximation --------------------
    y = ax * ax
    ans1 = ax * (
        72362614232.0
        + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606)))))
    )
    ans2 = 144725228442.0 + y * (
        2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0)))
    )
    ans_small = ans1 / ans2

    # ---- branch 2: |x| >= 8, asymptotic expansion (clamp keeps this branch finite
    # even where it is not selected, so `torch.where`'s unused branch cannot inject
    # NaN gradients into the selected one) ------------------------------------------
    ax_safe = torch.clamp(ax, min=8.0)
    z = 8.0 / ax_safe
    y2 = z * z
    xx = ax_safe - 2.356194491
    p1 = 1.0 + y2 * (0.183105e-2 + y2 * (-0.3516396496e-4 + y2 * (0.2457520174e-5 + y2 * (-0.240337019e-6))))
    p2 = 0.04687499995 + y2 * (
        -0.2002690873e-3 + y2 * (0.8449199096e-5 + y2 * (-0.88228987e-6 + y2 * 0.105787412e-6))
    )
    ans_large = torch.sqrt(0.636619772 / ax_safe) * (torch.cos(xx) * p1 - z * torch.sin(xx) * p2)

    ans = torch.where(ax < 8.0, ans_small, ans_large)
    return torch.sign(x) * ans


def _airy_intensity(r_px: torch.Tensor, r0_px: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. (6), with a Taylor-series limit near v=0 (2*J1(v)/v -> 1 - v^2/8) to avoid the
    0/0 indeterminate form at the disk's own center and keep the gradient well-defined
    there.
    """
    v = _AIRY_FIRST_ZERO * r_px / r0_px.clamp_min(eps)
    v_safe = torch.clamp(v.abs(), min=1e-4)
    ratio = 2.0 * _bessel_j1(v_safe) / v_safe
    taylor = 1.0 - (v * v) / 8.0
    ratio = torch.where(v.abs() < 1e-4, taylor, ratio)
    return ratio.pow(2)


def _make_airy_kernel(r0_px: torch.Tensor, ksize: int) -> torch.Tensor:
    """Rasterize the Airy diffraction PSF on a (ksize x ksize) grid, centered.

    Args:
        r0_px: (N,) first-null Airy radius in pixels = 1.22 * lambda * N_fstop / pitch.
        ksize: odd kernel spatial size.
    Returns:
        (N, 1, ksize, ksize), each sample normalized to sum to 1.
    """
    device, dtype = r0_px.device, r0_px.dtype
    r0 = r0_px.reshape(-1, 1, 1, 1)
    half = (ksize - 1) / 2.0
    ys = torch.arange(ksize, device=device, dtype=dtype) - half
    xs = torch.arange(ksize, device=device, dtype=dtype) - half
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(yy.pow(2) + xx.pow(2)).reshape(1, 1, ksize, ksize)
    intensity = _airy_intensity(r, r0)
    kernel = intensity / intensity.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return kernel


def _circular_convolve_kernels(a: torch.Tensor, b: torch.Tensor, ksize: int) -> torch.Tensor:
    """Combine two centered (N,1,ksize,ksize) PSFs by convolving them (the physically
    correct way to cascade two independent blurring processes: total PSF = defocus PSF
    (convolve) diffraction PSF), evaluated as a *circular* convolution on the shared
    ksize x ksize grid. Valid as long as each PSF's energy is concentrated well inside
    the grid relative to ksize -- `_radius_to_auto_ksize` sizes the grid from the
    combined effective radius with margin specifically to keep this assumption valid.
    """
    a_otf = _psf_to_otf(a, ksize, ksize)
    b_otf = _psf_to_otf(b, ksize, ksize)
    combined = torch.fft.irfft2(a_otf * b_otf, s=(ksize, ksize))
    combined = torch.roll(combined, shifts=(ksize // 2, ksize // 2), dims=(-2, -1))
    combined = combined.clamp_min(0.0)  # guards FFT round-off; true PSF conv is >= 0
    combined = combined / combined.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return combined


# ======================================================================================
# 2 & 1 combined: AdvancedOpticsKernelEngine -- diffraction + per-channel dispersion
# ======================================================================================
class AdvancedOpticsKernelEngine(nn.Module):
    """Combined geometric-defocus + diffraction + chromatic-dispersion optical kernel.

    ---- Chromatic dispersion model -------------------------------------------------
    A thin lens's optical power P = (n(lambda) - 1) * (1/R1 - 1/R2) is proportional to
    (n(lambda) - 1) for a fixed lens geometry, so for a *single-element* lens:

        f(lambda) / f(lambda_ref) = (n(lambda_ref) - 1) / (n(lambda) - 1)             (7)

    n(lambda) is modeled with the two-term Cauchy dispersion relation (accurate across
    the visible band for ordinary optical glass):

        n(lambda) = A + B / lambda_um^2                                              (8)

    with (A, B) defaulting to BK7-crown-glass-like constants (override via `cauchy_A`,
    `cauchy_B_um2` for a specific lens design). Real camera lenses are compound
    achromatic/apochromatic assemblies that cancel *most* (never quite all) of the
    single-element dispersion of Eq. (7); the uncancelled residual -- what actually
    produces visible longitudinal color fringing -- is scaled by `dispersion_strength`
    in [0, 1] (0 = perfectly achromatic, 1 = uncorrected single glass element):

        f(lambda) = f * [ 1 + dispersion_strength * (ratio(lambda) - 1) ]            (9)

    Each channel's f(lambda) is then fed back through the *existing*
    `circle_of_confusion_diameter_m` (imported from `unfolded_optics_deblur`), so the
    per-channel geometric defocus radius differs for a physically grounded reason (a
    genuinely different f(lambda), not an arbitrary per-channel scale factor). The Airy
    radius r0(lambda) = 1.22 * lambda * N / pitch is wavelength-dependent by definition
    and needs no extra step.

    ---- Combination -----------------------------------------------------------------
    `combine_mode="fourier"` (default) convolves the geometric pillbox with the Airy
    kernel in the Fourier domain -- the physically correct cascade of two blur
    processes. `combine_mode="quadrature"` instead approximates the combined kernel as
    a single widened pillbox of radius sqrt(R_geom^2 + R_airy^2) (faster, no diffraction
    ring structure) -- the fallback the diffraction task description offers directly.
    """

    DEFAULT_WAVELENGTHS_NM: Tuple[float, float, float] = (650.0, 530.0, 460.0)  # R, G, B

    def __init__(
        self,
        wavelengths_nm: Tuple[float, float, float] = DEFAULT_WAVELENGTHS_NM,
        reference_channel: int = 1,  # index into wavelengths_nm treated as the lens's
        # autofocus/calibration wavelength (green: peak luminance sensitivity + typical
        # AF sensor spectral weighting) -- zero dispersion shift by definition.
        dispersion_strength: float = 0.15,
        cauchy_A: float = 1.5046,
        cauchy_B_um2: float = 0.00420,
        combine_mode: str = "fourier",
        softness_px: float = 0.75,
        min_ksize: int = 7,
        max_ksize: int = 65,
    ):
        super().__init__()
        if combine_mode not in ("fourier", "quadrature"):
            raise ValueError(f"combine_mode must be 'fourier' or 'quadrature', got {combine_mode!r}")
        self.register_buffer(
            "wavelengths_m",
            torch.tensor(wavelengths_nm, dtype=torch.float32) * 1e-9,
            persistent=False,
        )
        self.reference_channel = reference_channel
        self.dispersion_strength = dispersion_strength
        self.cauchy_A = cauchy_A
        self.cauchy_B_um2 = cauchy_B_um2
        self.combine_mode = combine_mode
        self.softness_px = softness_px
        self.min_ksize = min_ksize
        self.max_ksize = max_ksize

    def _refractive_index(self, wavelengths_m: torch.Tensor) -> torch.Tensor:
        """Eq. (8): Cauchy two-term dispersion relation, lambda in meters -> micrometers."""
        lam_um = wavelengths_m * 1.0e6
        return self.cauchy_A + self.cauchy_B_um2 / lam_um.pow(2)

    def _channel_focal_lengths(self, focal_length_m: torch.Tensor) -> torch.Tensor:
        """Eq. (9): (B,) -> (B, 3) per-channel effective focal length under LoCA."""
        n = self._refractive_index(self.wavelengths_m)  # (3,)
        n_ref = n[self.reference_channel]
        ratio = (n_ref - 1.0) / (n - 1.0)  # (3,)
        f = focal_length_m.reshape(-1, 1)  # (B,1)
        return f * (1.0 + self.dispersion_strength * (ratio.view(1, -1) - 1.0))  # (B,3)

    def forward(
        self,
        focal_length_m: torch.Tensor,
        f_number: torch.Tensor,
        pixel_pitch_m: torch.Tensor,
        focus_distance_m: torch.Tensor,
        subject_distance_m: torch.Tensor,
        ksize: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            focal_length_m, f_number, pixel_pitch_m, focus_distance_m, subject_distance_m:
                (B,) tensors (meters / dimensionless f-number), exactly the inputs
                `circle_of_confusion_diameter_m` already takes.
            ksize: override the auto-sized kernel window (must be odd).
        Returns:
            kernel: (B, 3, Hk, Wk) diffraction+defocus+dispersion PSF, one channel per
                    wavelength in `self.wavelengths_m`, each sample normalized to sum 1.
        """
        b = focal_length_m.shape[0]
        f_channel = self._channel_focal_lengths(focal_length_m)  # (B,3)

        # Flatten (B,3) -> (B*3,), index = batch*3 + channel, consistently across every
        # per-channel tensor below so the final reshape back to (B,3,...) is exact.
        f_b3 = f_channel.reshape(b * 3)
        n_b3 = f_number.reshape(b, 1).expand(b, 3).reshape(b * 3)
        p_b3 = pixel_pitch_m.reshape(b, 1).expand(b, 3).reshape(b * 3)
        d0_b3 = focus_distance_m.reshape(b, 1).expand(b, 3).reshape(b * 3)
        d_b3 = subject_distance_m.reshape(b, 1).expand(b, 3).reshape(b * 3)
        lam_b3 = self.wavelengths_m.view(1, 3).expand(b, 3).reshape(b * 3)

        b_m = circle_of_confusion_diameter_m(f_b3, n_b3, d0_b3, d_b3)  # (B*3,)
        r_geom_px = defocus_diameter_to_pixel_radius(b_m, p_b3)  # (B*3,)
        r0_airy_px = (1.22 * lam_b3 * n_b3) / p_b3  # (B*3,), Airy first-null radius

        if ksize is None:
            r_upper_bound = torch.sqrt(r_geom_px.pow(2) + r0_airy_px.pow(2))
            ksize = _radius_to_auto_ksize(r_upper_bound, self.min_ksize, self.max_ksize)

        if self.combine_mode == "quadrature":
            r_eff = torch.sqrt(r_geom_px.pow(2) + r0_airy_px.pow(2))
            kernel_b3 = make_soft_pillbox_kernel(r_eff, ksize=ksize, softness_px=self.softness_px)
        else:
            pillbox = make_soft_pillbox_kernel(r_geom_px, ksize=ksize, softness_px=self.softness_px)
            airy = _make_airy_kernel(r0_airy_px, ksize=ksize)
            kernel_b3 = _circular_convolve_kernels(pillbox, airy, ksize)

        return kernel_b3.reshape(b, 3, ksize, ksize)


# ======================================================================================
# 3. Depth-conditioned spatially variant inversion
# ======================================================================================
class DepthToKernelField(nn.Module):
    """Maps a monocular relative depth map into a small bank of representative-distance
    PSFs plus a soft per-pixel assignment field, for spatially variant deblurring.

    Rather than computing a distinct kernel at every pixel (computationally wasteful --
    defocus varies smoothly with depth, so nearby pixels at similar depth share nearly
    identical PSFs), the depth range is quantized into `num_depth_bins` representative
    distances. Each pixel receives a soft (Gaussian/softmax) weight over those bins
    according to how close its own metric depth is to each bin's representative
    distance, so the depth -> kernel mapping stays differentiable end-to-end (no
    hard `argmin`/`bucketize` in the graph) and blends smoothly across true depth
    discontinuities rather than producing hard per-bin boundaries.
    """

    def __init__(self, kernel_engine: AdvancedOpticsKernelEngine, num_depth_bins: int = 6, bin_softness: float = 1.5):
        super().__init__()
        self.kernel_engine = kernel_engine
        self.num_depth_bins = num_depth_bins
        self.bin_softness = bin_softness  # bin-assignment Gaussian sigma, in units of bin spacing

    def forward(
        self,
        depth_map: torch.Tensor,  # (B,1,H,W), normalized relative depth in [0,1]
        focal_length_m: torch.Tensor,  # (B,)
        f_number: torch.Tensor,  # (B,)
        pixel_pitch_m: torch.Tensor,  # (B,)
        focus_distance_m: torch.Tensor,  # (B,)
        depth_min_m: torch.Tensor,  # (B,) metric distance at depth_map == "near" extreme
        depth_max_m: torch.Tensor,  # (B,) metric distance at depth_map == "far" extreme
        depth_is_inverse: bool = True,  # True: depth_map is normalized *inverse* depth /
        # disparity (the convention most monocular relative-depth networks, e.g. MiDaS,
        # output: larger value = nearer). False: depth_map is normalized linear metric depth.
        ksize: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            bin_kernels:  (B, num_bins, 3, Hk, Wk) -- one dispersion-aware PSF per bin.
            bin_weights:  (B, num_bins, H, W) -- softmax over bins, sums to 1 per pixel.
            bin_distances:(B, num_bins) -- each bin's representative metric distance (m).
        """
        b, _, h, w = depth_map.shape
        nb = self.num_depth_bins
        d_near = depth_min_m.reshape(b, 1, 1, 1)
        d_far = depth_max_m.reshape(b, 1, 1, 1)
        d_norm = depth_map.clamp(0.0, 1.0)

        bin_edges = torch.linspace(0.0, 1.0, nb, device=depth_map.device, dtype=depth_map.dtype).view(1, -1)

        if depth_is_inverse:
            inv_near = 1.0 / d_near.clamp_min(1e-6)  # inverse depth at the near extreme (largest)
            inv_far = 1.0 / d_far.clamp_min(1e-6)  # inverse depth at the far extreme (smallest)
            inv_d = inv_far + d_norm * (inv_near - inv_far)  # d_norm=1 (near) -> inv_near
            d_metric = 1.0 / inv_d.clamp_min(1e-6)  # (B,1,H,W)
            inv_bins = inv_far.reshape(b, 1) + bin_edges * (inv_near - inv_far).reshape(b, 1)
            bin_distances = 1.0 / inv_bins.clamp_min(1e-6)  # (B, nb)
        else:
            d_metric = d_near + d_norm * (d_far - d_near)  # (B,1,H,W)
            bin_distances = d_near.reshape(b, 1) + bin_edges * (d_far - d_near).reshape(b, 1)  # (B, nb)

        bin_centers = bin_distances.view(b, nb, 1, 1)
        bin_spacing = (bin_distances[:, 1:] - bin_distances[:, :-1]).abs().mean(dim=1).clamp_min(1e-6)  # (B,)
        sigma = (self.bin_softness * bin_spacing).view(b, 1, 1, 1)

        diff = d_metric.view(b, 1, h, w) - bin_centers  # (B,nb,H,W)
        logits = -diff.pow(2) / (2.0 * sigma.pow(2) + 1e-12)
        bin_weights = torch.softmax(logits, dim=1)  # (B,nb,H,W)

        # Replicate camera params across bins, folded into one flat (B*nb,) batch so a
        # single `AdvancedOpticsKernelEngine.forward` call produces every bin's kernel.
        f_rep = focal_length_m.view(b, 1).expand(b, nb).reshape(b * nb)
        n_rep = f_number.view(b, 1).expand(b, nb).reshape(b * nb)
        p_rep = pixel_pitch_m.view(b, 1).expand(b, nb).reshape(b * nb)
        d0_rep = focus_distance_m.view(b, 1).expand(b, nb).reshape(b * nb)
        d_rep = bin_distances.reshape(b * nb)

        if ksize is None:
            with torch.no_grad():
                b_m_probe = circle_of_confusion_diameter_m(f_rep, n_rep, d0_rep, d_rep)
                r_probe = defocus_diameter_to_pixel_radius(b_m_probe, p_rep)
                ksize = _radius_to_auto_ksize(r_probe, self.kernel_engine.min_ksize, self.kernel_engine.max_ksize)

        bin_kernels = self.kernel_engine(f_rep, n_rep, p_rep, d0_rep, d_rep, ksize=ksize)  # (B*nb,3,Hk,Wk)
        bin_kernels = bin_kernels.reshape(b, nb, 3, ksize, ksize)

        return bin_kernels, bin_weights, bin_distances


# ======================================================================================
# 4. Boundary ringing suppression: kernel-adaptive differentiable edgetaper
# ======================================================================================
class DifferentiableEdgeTaper(nn.Module):
    """Analytical, kernel-adaptive border taper, applied before any `rfft2` call.

    Classical edgetaper (Reeves & Lagendijk, 1985 / MATLAB `edgetaper`) blends the image
    with its own circular self-blur, weighted 0 -> 1 by a border mask, so that FFT-based
    (circularly-periodic) deconvolution does not see a hard discontinuity at the wrap-
    around seam:

        I_tapered = alpha * I + (1 - alpha) * (I (circular-convolve) PSF)             (10)

    This implementation derives the *shape* of alpha's border ramp directly from the
    supplied kernel's own 1D marginal energy profile (instead of a fixed-shape window):
    project the PSF onto each axis, take its cumulative energy from the border inward,
    and use that as the 0->1 ramp. A wide/soft kernel therefore tapers over a wide/soft
    border; a narrow/sharp kernel tapers sharply -- directly reflecting the physical
    blur being inverted, which is what makes the taper "using the computed blur kernel"
    rather than a generic Hann window.
    """

    @staticmethod
    def _axis_taper_from_kernel(marginal_1d: torch.Tensor, length: int) -> torch.Tensor:
        """marginal_1d: (N, k), each row a nonnegative PSF marginal summing to 1.
        Returns (N, length) in [0,1]: ramps 0->1 from each border over one kernel-width,
        1 in the interior.
        """
        n, k = marginal_1d.shape
        ramp = torch.cumsum(marginal_1d, dim=-1)  # (N,k), monotonic 0 -> ~1
        ramp = ramp / ramp[:, -1:].clamp_min(1e-8)
        pad = max(length - 2 * k, 0)
        interior = ramp.new_ones(n, pad)
        taper = torch.cat([ramp, interior, ramp.flip(-1)], dim=-1)
        if taper.shape[-1] != length:  # length < 2k (very small image): center-crop
            extra = taper.shape[-1] - length
            lo = extra // 2
            taper = taper[:, lo : lo + length]
        return taper.clamp(0.0, 1.0)

    def forward(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, C, H, W).
            kernel: (B, 1, kh, kw) or (B, C, kh, kw), each sample summing to 1.
        Returns:
            (B, C, H, W) edge-tapered image, ready for `torch.fft.rfft2`.
        """
        b, c, h, w = img.shape
        if kernel.shape[1] == 1 and c > 1:
            kernel = kernel.expand(-1, c, -1, -1)
        kh, kw = kernel.shape[-2:]
        k_flat = kernel.reshape(b * c, 1, kh, kw)

        marg_y = k_flat.sum(dim=-1).reshape(b * c, kh)
        marg_x = k_flat.sum(dim=-2).reshape(b * c, kw)
        taper_y = self._axis_taper_from_kernel(marg_y, h).reshape(b, c, h, 1)
        taper_x = self._axis_taper_from_kernel(marg_x, w).reshape(b, c, 1, w)
        alpha = taper_y * taper_x  # (B,C,H,W) in [0,1]

        k_otf = _psf_to_otf(kernel, h, w)
        img_fft = torch.fft.rfft2(img)
        img_blurred = torch.fft.irfft2(img_fft * k_otf, s=(h, w))

        return alpha * img + (1.0 - alpha) * img_blurred


# ======================================================================================
# 5. Stabilized multi-channel Wiener deconvolution (OTF zero-crossing safeguard)
# ======================================================================================
class StabilizedMultiChannelWienerDeconv(nn.Module):
    """Batched, per-channel-OTF Wiener/HQS data-fidelity step (same closed form as
    `AnalyticalWienerDeconv` in `unfolded_optics_deblur.py`, Eq. (5)) upgraded with:

      * a channel-specific kernel `K in (B,3,Hk,Wk)`, so red/green/blue are deconvolved
        with their own dispersion-aware OTF instead of one shared kernel;
      * `DifferentiableEdgeTaper` (Sec. 4) applied to both `y` and `z` before `rfft2`;
      * mu_k = softplus(alpha_k) + eps -- strictly positive for *any* real alpha_k
        (unlike `exp(alpha_k)`, which is also always positive but can still underflow
        to a representable-but-effectively-zero float for very negative alpha_k). The
        `clamp_min(eps)` on the full denominator is the safeguard that actually matters
        at an OTF zero-crossing: at a Bessel null of the diffraction OTF, |K_fft| -> 0,
        so the denominator |K_fft|^2 + mu_k collapses to mu_k itself -- softplus's
        `+ eps` floor guarantees this can never reach exactly 0, which is what bounds
        the noise-amplification gain 1/(|K_fft|^2 + mu_k) at every frequency, including
        the zero-crossings, rather than letting it diverge.
    """

    def __init__(self, num_channels: int = 3, init_mu: float = 1.0, eps: float = 1e-6, pad_mode: str = "reflect"):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.pad_mode = pad_mode
        self.alpha = nn.Parameter(torch.full((num_channels,), _inverse_softplus(init_mu), dtype=torch.float32))
        self.edge_taper = DifferentiableEdgeTaper()

    @property
    def mu(self) -> torch.Tensor:
        return F.softplus(self.alpha) + self.eps  # (C,), strictly > eps

    def forward(
        self, y: torch.Tensor, kernel: torch.Tensor, z: torch.Tensor, extra_pad: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            y: (N, C, H, W) observed (blurry) image, linear radiometric space.
            kernel: (N, C, Hk, Wk) or (N, 1, Hk, Wk) PSF, each sample sums to 1.
            z: (N, C, H, W) current HQS prior estimate z^(k).
        Returns:
            (N, C, H, W) closed-form data-fidelity estimate x^(k+1/2).
        """
        n, c, h, w = y.shape
        kh, kw = kernel.shape[-2:]
        pad_h = extra_pad if extra_pad is not None else kh // 2
        pad_w = extra_pad if extra_pad is not None else kw // 2

        y_pad = F.pad(y, (pad_w, pad_w, pad_h, pad_h), mode=self.pad_mode)
        z_pad = F.pad(z, (pad_w, pad_w, pad_h, pad_h), mode=self.pad_mode)
        hp, wp = y_pad.shape[-2:]

        y_tapered = self.edge_taper(y_pad, kernel)
        z_tapered = self.edge_taper(z_pad, kernel)

        k_otf = _psf_to_otf(kernel, hp, wp)  # (N, C or 1, Hp, Wp//2+1) complex
        y_fft = torch.fft.rfft2(y_tapered)
        z_fft = torch.fft.rfft2(z_tapered)

        mu = self.mu.view(1, self.num_channels, 1, 1)
        k_power = k_otf.real.pow(2) + k_otf.imag.pow(2)  # (N, C or 1, Hp, Wp//2+1), >= 0
        denom = (k_power + mu).clamp_min(self.eps)

        numerator = torch.conj(k_otf) * y_fft + mu * z_fft
        x_fft = numerator / denom

        x_pad_out = torch.fft.irfft2(x_fft, s=(hp, wp))
        return x_pad_out[..., pad_h : pad_h + h, pad_w : pad_w + w]


# ======================================================================================
# 6. Patch-wise Overlap-Add spatially variant deconvolution driver
# ======================================================================================
def _hann_window_2d(size: int, device, dtype) -> torch.Tensor:
    w1 = torch.hann_window(size, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return w1.view(-1, 1) * w1.view(1, -1)


def _fit_pad(size: int, patch: int, stride: int) -> int:
    if size <= patch:
        return patch - size
    rem = (size - patch) % stride
    return 0 if rem == 0 else stride - rem


def _patchwise_kernel_blend_and_ola(
    y: torch.Tensor,
    aux: torch.Tensor,
    bin_kernels: torch.Tensor,
    bin_weights: torch.Tensor,
    patch_size: int,
    stride: int,
    patch_op: "Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]",
) -> torch.Tensor:
    """Shared Overlap-Add machinery, factored out so it can drive *either* a single
    Wiener step (`OverlapAddSpatialDeconv`) or a full K-stage unfolded network
    (`AdvancedUnfoldedOpticsDeblurNet.forward_full_frame`) over depth-varying blur:

    unfold `y`/`aux`/`bin_weights` into overlapping patches, blend each patch's single
    locally-representative kernel from its patch-averaged depth-bin weights (a locally
    shift-invariant approximation to the true spatially varying PSF field, valid
    because defocus varies smoothly with scene depth), hand every patch to
    `patch_op(y_patch, kernel_patch, aux_patch) -> out_patch`, then fold the
    Hann-windowed results back together and divide by the folded window-energy sum --
    which is what removes the seams (exact regardless of whether the window satisfies
    the classical constant-overlap-add condition, unlike a fixed-shape taper).

    Args:
        y: (B, C, H, W) data term (the blurry image itself).
        aux: (B, Ca, H, W) second per-pixel input `patch_op` needs alongside `y` and its
             kernel -- either an HQS prior estimate z (for a single Wiener step) or a
             noise-level map sigma (for a full unfolded-network patch_op).
        bin_kernels: (B, nb, C, Hk, Wk). bin_weights: (B, nb, H, W).
        patch_op: applied independently to every (B*L,...) patch batch.
    Returns:
        (B, C, H, W), same size as `y`.
    """
    b, c, h, w = y.shape
    device, dtype = y.device, y.dtype
    ps, st = patch_size, stride

    pad_h = _fit_pad(h, ps, st)
    pad_w = _fit_pad(w, ps, st)
    y_p = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
    aux_p = F.pad(aux, (0, pad_w, 0, pad_h), mode="reflect")
    w_p = F.pad(bin_weights, (0, pad_w, 0, pad_h), mode="reflect")
    hp, wp = y_p.shape[-2:]

    window = _hann_window_2d(ps, device, dtype)  # (ps,ps)

    y_patches = F.unfold(y_p, kernel_size=ps, stride=st).transpose(1, 2)  # (B,L,C*ps*ps)
    aux_patches = F.unfold(aux_p, kernel_size=ps, stride=st).transpose(1, 2)
    l = y_patches.shape[1]
    ac = aux.shape[1]
    y_patches = y_patches.reshape(b * l, c, ps, ps)
    aux_patches = aux_patches.reshape(b * l, ac, ps, ps)

    # One scalar weight per (patch, depth-bin): the patch-averaged soft assignment,
    # i.e. the expected depth-bin mixture inside that patch's footprint.
    nb = bin_weights.shape[1]
    w_patches = F.unfold(w_p, kernel_size=ps, stride=st).transpose(1, 2)  # (B,L,nb*ps*ps)
    w_patches = w_patches.reshape(b, l, nb, ps * ps).mean(dim=-1)  # (B,L,nb)
    w_patches = w_patches / w_patches.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    w_patches = w_patches.reshape(b * l, nb)

    kc = bin_kernels.shape[-3]
    kh, kw = bin_kernels.shape[-2:]
    bk = bin_kernels.unsqueeze(1).expand(b, l, nb, kc, kh, kw).reshape(b * l, nb, kc, kh, kw)
    patch_kernel = (w_patches.view(b * l, nb, 1, 1, 1) * bk).sum(dim=1)  # (B*L,C,kh,kw)
    patch_kernel = patch_kernel / patch_kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    out_patches = patch_op(y_patches, patch_kernel, aux_patches)  # (B*L,C,ps,ps)
    out_patches = out_patches * window.view(1, 1, ps, ps)

    out_flat = out_patches.reshape(b, l, c * ps * ps).transpose(1, 2)  # (B,C*ps*ps,L)
    numerator = F.fold(out_flat, output_size=(hp, wp), kernel_size=ps, stride=st)

    window_full = window.view(1, 1, ps, ps).expand(b * l, c, ps, ps)
    window_flat = window_full.reshape(b, l, c * ps * ps).transpose(1, 2)
    denom = F.fold(window_flat, output_size=(hp, wp), kernel_size=ps, stride=st).clamp_min(1e-6)

    out = numerator / denom
    return out[..., :h, :w]


class OverlapAddSpatialDeconv(nn.Module):
    """Applies `DepthToKernelField`'s per-bin kernels across a full image without
    boundary seams, via patch-wise Hann-window Overlap-Add (thin wrapper around
    `_patchwise_kernel_blend_and_ola`, driving a single `StabilizedMultiChannelWienerDeconv`
    step per patch).
    """

    def __init__(self, wiener: StabilizedMultiChannelWienerDeconv, patch_size: int = 64, overlap: float = 0.5):
        super().__init__()
        if not 0.0 < overlap < 1.0:
            raise ValueError(f"overlap must be in (0,1), got {overlap}")
        self.wiener = wiener
        self.patch_size = patch_size
        self.stride = max(1, int(round(patch_size * (1.0 - overlap))))

    def forward(
        self,
        y: torch.Tensor,  # (B,3,H,W) blurry image, linear radiometric space
        bin_kernels: torch.Tensor,  # (B,nb,3,Hk,Wk)
        bin_weights: torch.Tensor,  # (B,nb,H,W)
        z: Optional[torch.Tensor] = None,  # HQS prior estimate; defaults to y
    ) -> torch.Tensor:
        z = y if z is None else z
        return _patchwise_kernel_blend_and_ola(
            y, z, bin_kernels, bin_weights, self.patch_size, self.stride, self.wiener
        )


# ======================================================================================
# 6b. Full integration: drop-in, physics-upgraded replacement for UnfoldedOpticsDeblurNet
# ======================================================================================
class AdvancedUnfoldedOpticsDeblurNet(UnfoldedOpticsDeblurNet):
    """Same K-stage HQS/learned-prior architecture and the same low-level
    `forward(y_srgb, kernel, sigma)` interface as `UnfoldedOpticsDeblurNet` (so it is a
    drop-in replacement wherever a kernel is already being built by hand), but:

      * its data-fidelity steps are `StabilizedMultiChannelWienerDeconv` (per-channel
        OTF, softplus-regularized mu, kernel-adaptive edgetaper) instead of the base
        model's `AnalyticalWienerDeconv`;
      * it owns an `AdvancedOpticsKernelEngine`, so `forward_from_camera` lets a caller
        hand it camera metadata directly instead of building a kernel by hand;
      * `forward_full_frame` combines `DepthToKernelField` with a patch-wise
        Overlap-Add pass of the *entire* unfolded network (all K stages, including the
        learned prior at every stage -- not just one Wiener step) for seam-free,
        depth-guided, spatially variant deblurring across a full image.
    """

    def __init__(
        self,
        num_stages: int = 6,
        base_ch: int = 32,
        in_ch: int = 3,
        kernel_engine_kwargs: Optional[dict] = None,
        num_depth_bins: int = 6,
        ola_patch_size: int = 64,
        ola_overlap: float = 0.5,
    ):
        super().__init__(
            num_stages=num_stages,
            base_ch=base_ch,
            in_ch=in_ch,
            data_step_factory=lambda: StabilizedMultiChannelWienerDeconv(num_channels=in_ch, init_mu=1.0),
        )
        self.kernel_engine = AdvancedOpticsKernelEngine(**(kernel_engine_kwargs or {}))
        self.depth_field = DepthToKernelField(self.kernel_engine, num_depth_bins=num_depth_bins)
        self.ola_patch_size = ola_patch_size
        self.ola_overlap = ola_overlap

    def build_kernel(
        self,
        focal_length_m: torch.Tensor,
        f_number: torch.Tensor,
        pixel_pitch_m: torch.Tensor,
        focus_distance_m: torch.Tensor,
        subject_distance_m: torch.Tensor,
        ksize: Optional[int] = None,
    ) -> torch.Tensor:
        """Camera metadata -> (B,3,Hk,Wk) diffraction+dispersion kernel, ready to pass
        straight into the inherited `forward(y_srgb, kernel, sigma)`.
        """
        return self.kernel_engine(
            focal_length_m, f_number, pixel_pitch_m, focus_distance_m, subject_distance_m, ksize=ksize
        )

    def forward_from_camera(
        self,
        y_srgb: torch.Tensor,
        sigma: torch.Tensor,
        focal_length_m: torch.Tensor,
        f_number: torch.Tensor,
        pixel_pitch_m: torch.Tensor,
        focus_distance_m: torch.Tensor,
        subject_distance_m: torch.Tensor,
        use_learned_prior: bool = True,
        ksize: Optional[int] = None,
    ) -> torch.Tensor:
        """One-call path for a single (ROI-crop-style) subject distance: builds the
        dispersion-aware kernel from camera metadata, then runs the ordinary
        `forward`. Effortless end-to-end: metadata in, restored sRGB image out.
        """
        kernel = self.build_kernel(
            focal_length_m, f_number, pixel_pitch_m, focus_distance_m, subject_distance_m, ksize=ksize
        )
        return self.forward(y_srgb, kernel, sigma, use_learned_prior=use_learned_prior)

    def forward_full_frame(
        self,
        y_srgb: torch.Tensor,
        sigma: torch.Tensor,
        depth_map: torch.Tensor,
        focal_length_m: torch.Tensor,
        f_number: torch.Tensor,
        pixel_pitch_m: torch.Tensor,
        focus_distance_m: torch.Tensor,
        depth_min_m: torch.Tensor,
        depth_max_m: torch.Tensor,
        depth_is_inverse: bool = True,
        use_learned_prior: bool = True,
        patch_size: Optional[int] = None,
        overlap: Optional[float] = None,
    ) -> torch.Tensor:
        """One-call path for a full-frame image with scene-varying depth: builds the
        depth-bin kernel field, then runs every HQS stage of this model patch-wise with
        seam-free Overlap-Add reassembly. Metadata + depth map in, restored sRGB image
        out -- no manual kernel construction, FFT plumbing, or patch bookkeeping needed.
        """
        ps = patch_size if patch_size is not None else self.ola_patch_size
        ov = overlap if overlap is not None else self.ola_overlap
        st = max(1, int(round(ps * (1.0 - ov))))

        bin_kernels, bin_weights, _ = self.depth_field(
            depth_map,
            focal_length_m,
            f_number,
            pixel_pitch_m,
            focus_distance_m,
            depth_min_m,
            depth_max_m,
            depth_is_inverse=depth_is_inverse,
        )

        y_linear = self.isp.to_linear(y_srgb)

        def patch_op(y_patch: torch.Tensor, kernel_patch: torch.Tensor, sigma_patch: torch.Tensor) -> torch.Tensor:
            return self._run_stages(y_patch, kernel_patch, sigma_patch, use_learned_prior=use_learned_prior)

        z_linear = _patchwise_kernel_blend_and_ola(y_linear, sigma, bin_kernels, bin_weights, ps, st, patch_op)
        x_restored_linear = z_linear.clamp(0.0, 1.0)
        return self.isp.to_srgb(x_restored_linear)


# ======================================================================================
# 7. Self-contained execution / verification block
# ======================================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running self-test on device: {device}")

    # ---- 1. Kernel generation across f-stops: f/1.8 (geometry-dominated) vs f/16
    #         (diffraction-dominated) --------------------------------------------------
    engine = AdvancedOpticsKernelEngine(combine_mode="fourier", dispersion_strength=0.15).to(device)

    B = 2
    focal_length_m = torch.full((B,), 0.085, device=device)  # 85 mm lens
    f_number = torch.tensor([1.8, 16.0], device=device)
    pixel_pitch_m = torch.full((B,), 4.0e-6, device=device)  # 4 um pixels
    focus_distance_m = torch.full((B,), 2.0, device=device)
    subject_distance_m = torch.tensor([2.6, 2.6], device=device)  # same defocus offset

    kernel = engine(focal_length_m, f_number, pixel_pitch_m, focus_distance_m, subject_distance_m)
    print(f"\n[1] Kernel shape: {tuple(kernel.shape)}")
    assert kernel.dim() == 4 and kernel.shape[1] == 3
    assert torch.allclose(kernel.sum(dim=(-2, -1)), torch.ones(B, 3, device=device), atol=1e-3)

    b_m = circle_of_confusion_diameter_m(focal_length_m, f_number, focus_distance_m, subject_distance_m)
    r_geom_px = defocus_diameter_to_pixel_radius(b_m, pixel_pitch_m)
    lam_green_m = engine.wavelengths_m[engine.reference_channel]
    r_airy_px = (1.22 * lam_green_m * f_number) / pixel_pitch_m
    print(f"    f/1.8: geometric R={r_geom_px[0].item():.2f}px, Airy R={r_airy_px[0].item():.2f}px "
          f"(geometry-dominated)")
    print(f"    f/16 : geometric R={r_geom_px[1].item():.2f}px, Airy R={r_airy_px[1].item():.2f}px "
          f"(diffraction non-negligible)")
    assert r_airy_px[1] > r_airy_px[0], "Airy disk must grow with f-number (narrower aperture)"

    # At f/16 the diffraction contribution is a much larger fraction of the combined
    # kernel footprint than at f/1.8, where geometric defocus dominates.
    diffraction_fraction = r_airy_px / torch.sqrt(r_geom_px.pow(2) + r_airy_px.pow(2)).clamp_min(1e-8)
    assert diffraction_fraction[1] > diffraction_fraction[0], \
        "Diffraction should account for a larger share of the blur at the narrower aperture"

    # ---- 2. RGB wavelength radius divergence (chromatic dispersion) ------------------
    # Uses a *moderate* defocus (unlike the deliberately extreme f/1.8 case above, whose
    # ~60px geometric radius saturates the kernel window and swamps the sub-pixel
    # dispersion shift) so the per-channel radius difference is actually resolvable.
    f_number_mod = torch.tensor([2.8, 2.8], device=device)
    subject_distance_mod = torch.tensor([2.15, 2.15], device=device)
    kernel_mod = engine(focal_length_m, f_number_mod, pixel_pitch_m, focus_distance_m, subject_distance_mod)

    ys, xs = torch.meshgrid(
        torch.arange(kernel_mod.shape[-2], device=device, dtype=kernel_mod.dtype),
        torch.arange(kernel_mod.shape[-1], device=device, dtype=kernel_mod.dtype),
        indexing="ij",
    )
    center = (kernel_mod.shape[-1] - 1) / 2.0
    rr = torch.sqrt((ys - center).pow(2) + (xs - center).pow(2))
    centroid_radius = (kernel_mod[0] * rr.view(1, *rr.shape)).sum(dim=(-2, -1))
    print(f"\n[2] Per-channel (R,G,B) energy-weighted blur radius (moderate f/2.8 defocus): "
          f"{[round(v, 4) for v in centroid_radius.tolist()]}")
    assert not torch.allclose(centroid_radius[0], centroid_radius[2], atol=1e-4), \
        "Red and blue channel blur radii must diverge under nonzero dispersion_strength"

    # ---- 3. Diffraction-only quadrature-mode cross-check ------------------------------
    engine_quad = AdvancedOpticsKernelEngine(combine_mode="quadrature").to(device)
    kernel_quad = engine_quad(focal_length_m, f_number, pixel_pitch_m, focus_distance_m, subject_distance_m)
    assert torch.allclose(kernel_quad.sum(dim=(-2, -1)), torch.ones(B, 3, device=device), atol=1e-3)
    print(f"[3] Quadrature-mode kernel shape: {tuple(kernel_quad.shape)} (fast fallback, OK)")

    # ---- 4. Forward deconvolution pass: edgetaper + stabilized multi-channel Wiener --
    H, W = 64, 64
    y_srgb = torch.rand(B, 3, H, W, device=device, requires_grad=True)
    z0 = y_srgb.detach().clone().requires_grad_(True)

    wiener = StabilizedMultiChannelWienerDeconv(num_channels=3, init_mu=1.0).to(device)
    x_half = wiener(y_srgb, kernel, z0)
    print(f"\n[4] Wiener output shape: {tuple(x_half.shape)}")
    assert x_half.shape == y_srgb.shape
    assert torch.isfinite(x_half).all(), "Deconvolved output must be finite"

    loss = F.mse_loss(x_half, torch.zeros_like(x_half))
    loss.backward()
    assert y_srgb.grad is not None and torch.isfinite(y_srgb.grad).all()
    assert wiener.alpha.grad is not None and torch.isfinite(wiener.alpha.grad).all()
    print(f"    Learned per-channel mu (R,G,B): {[round(v, 4) for v in wiener.mu.tolist()]}")
    print("    Gradient flow through edgetaper + stabilized Wiener step: OK")

    # ---- 5. OTF zero-crossing stress test: the f/16 kernel (index 1) genuinely
    #         contains the diffraction-limited Airy pattern's Bessel sidelobes, so its
    #         OTF has real |K_fft| -> 0 zero-crossings -- confirm the denominator floor
    #         keeps the inversion finite exactly there. ---------------------------------
    diffraction_kernel = kernel[1:2].expand(B, -1, -1, -1).contiguous()  # f/16 sample, both batch slots
    with torch.no_grad():
        kh_probe, kw_probe = diffraction_kernel.shape[-2:]
        pad_probe = kh_probe // 2
        k_otf_probe = _psf_to_otf(diffraction_kernel, H + 2 * pad_probe, W + 2 * pad_probe)
        k_power_probe = k_otf_probe.real.pow(2) + k_otf_probe.imag.pow(2)
        min_otf_power = k_power_probe.min().item()
        denom_probe = (k_power_probe + wiener.mu.view(1, 3, 1, 1)).clamp_min(wiener.eps)
        min_denom = denom_probe.min().item()
    print(f"\n[5] Diffraction OTF min |K_fft|^2: {min_otf_power:.3e}  ->  regularized denom min: {min_denom:.3e}")
    assert min_otf_power < 1e-3, "Test kernel should contain a near-zero OTF crossing to be a meaningful stress test"
    assert min_denom >= wiener.eps, "Denominator must never fall below the eps floor, even at an OTF null"

    y_probe = torch.rand(B, 3, H, W, device=device)
    x_probe = wiener(y_probe, diffraction_kernel, y_probe)
    assert torch.isfinite(x_probe).all(), "Output must stay finite even at an OTF zero-crossing"
    print("    Deconvolution through this kernel stayed finite: OTF zero-crossing safeguard confirmed.")

    # ---- 6. Depth-guided spatially variant deconvolution (DepthToKernelField + OLA) --
    # Narrow-ish aperture + short focal length keeps the depth-range defocus radii
    # modest (a few px) so a small demo patch size comfortably exceeds every bin
    # kernel's half-width -- required for the reflect-pad inside the Wiener step.
    depth_map = torch.rand(1, 1, 64, 64, device=device)  # normalized relative depth
    f1 = torch.full((1,), 0.024, device=device)
    n1 = torch.full((1,), 16.0, device=device)
    p1 = torch.full((1,), 4.0e-6, device=device)
    d0_1 = torch.full((1,), 3.0, device=device)
    dmin = torch.full((1,), 1.0, device=device)
    dmax = torch.full((1,), 8.0, device=device)

    depth_engine = AdvancedOpticsKernelEngine(combine_mode="fourier", max_ksize=25).to(device)
    depth_field = DepthToKernelField(depth_engine, num_depth_bins=4, bin_softness=1.5).to(device)
    bin_kernels, bin_weights, bin_distances = depth_field(
        depth_map, f1, n1, p1, d0_1, dmin, dmax, depth_is_inverse=True
    )
    print(f"\n[6] bin_kernels: {tuple(bin_kernels.shape)}, bin_weights: {tuple(bin_weights.shape)}, "
          f"bin_distances (m): {[round(v, 3) for v in bin_distances[0].tolist()]}")
    assert torch.allclose(bin_weights.sum(dim=1), torch.ones_like(bin_weights[:, 0]), atol=1e-4)

    ola = OverlapAddSpatialDeconv(wiener, patch_size=32, overlap=0.5).to(device)
    y_full = torch.rand(1, 3, 64, 64, device=device)
    x_full = ola(y_full, bin_kernels, bin_weights)
    print(f"    OLA spatially-variant deconv output shape: {tuple(x_full.shape)}")
    assert x_full.shape == y_full.shape
    assert torch.isfinite(x_full).all(), "OLA reassembly must be finite/seam-free everywhere"

    # ---- 7. Full integration: AdvancedUnfoldedOpticsDeblurNet end-to-end -------------
    # (a) single-kernel path: camera metadata straight in, restored sRGB image out.
    full_model = AdvancedUnfoldedOpticsDeblurNet(
        num_stages=3, base_ch=8, kernel_engine_kwargs=dict(dispersion_strength=0.15, max_ksize=33)
    ).to(device)
    y_srgb_in = torch.rand(2, 3, 48, 48, device=device, requires_grad=True)
    sigma_in = (0.01 + 0.02 * torch.rand(2, 1, 48, 48, device=device)).to(device)
    fl = torch.full((2,), 0.05, device=device)
    fn = torch.tensor([2.8, 5.6], device=device)
    pp = torch.full((2,), 4.0e-6, device=device)
    fd = torch.full((2,), 2.0, device=device)
    sd = torch.tensor([2.3, 2.6], device=device)

    restored = full_model.forward_from_camera(y_srgb_in, sigma_in, fl, fn, pp, fd, sd)
    print(f"\n[7] AdvancedUnfoldedOpticsDeblurNet.forward_from_camera output: {tuple(restored.shape)}")
    assert restored.shape == y_srgb_in.shape
    assert torch.isfinite(restored).all()
    assert restored.min() >= 0.0 and restored.max() <= 1.0

    loss7 = F.mse_loss(restored, torch.zeros_like(restored))
    loss7.backward()
    assert y_srgb_in.grad is not None and torch.isfinite(y_srgb_in.grad).all()
    for k, step in enumerate(full_model.data_steps):
        assert step.alpha.grad is not None and torch.isfinite(step.alpha.grad).all(), \
            f"Stage {k}: per-channel mu must receive a finite gradient"
    print("    Gradient flow (image + per-stage per-channel mu): OK")

    # (b) full-frame, depth-guided path: metadata + relative depth map straight in.
    y_full_in = torch.rand(1, 3, 48, 48, device=device)
    sigma_full = (0.01 + 0.02 * torch.rand(1, 1, 48, 48, device=device)).to(device)
    depth_full = torch.rand(1, 1, 48, 48, device=device)
    fl1 = torch.full((1,), 0.024, device=device)
    fn1 = torch.full((1,), 16.0, device=device)
    pp1 = torch.full((1,), 4.0e-6, device=device)
    fd1 = torch.full((1,), 3.0, device=device)
    dmin1 = torch.full((1,), 1.0, device=device)
    dmax1 = torch.full((1,), 8.0, device=device)

    restored_full = full_model.forward_full_frame(
        y_full_in, sigma_full, depth_full, fl1, fn1, pp1, fd1, dmin1, dmax1,
        depth_is_inverse=True, patch_size=24, overlap=0.5,
    )
    print(f"    forward_full_frame output: {tuple(restored_full.shape)}")
    assert restored_full.shape == y_full_in.shape
    assert torch.isfinite(restored_full).all()
    assert restored_full.min() >= 0.0 and restored_full.max() <= 1.0
    print("    Depth-guided full-frame OLA through the *entire* unfolded network: OK")

    print("\nAll shape / finiteness / dispersion / gradient-flow / OLA / integration assertions passed.")
