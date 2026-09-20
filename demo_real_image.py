"""
Real-image test/demo for AdvancedUnfoldedOpticsDeblurNet.

There is no pretrained checkpoint (the model ships with random init only), so a single
random forward pass would not look "deblurred" -- it would just prove the plumbing
works. To actually *demonstrate* the physics-informed mechanism on a real photograph we
do the following, entirely with real image content:

  1. Take a real, public-domain test photograph (scikit-image's bundled "astronaut"
     sample -- a real photo of astronaut Eileen Collins, shipped with the library
     precisely for this kind of CV testing).
  2. Synthesize a physically realistic defocus + diffraction + chromatic-dispersion
     blurred, noisy observation from it, using `AdvancedOpticsKernelEngine` (i.e. the
     exact upgraded optics model the network now assumes) -- each RGB channel is
     convolved with its *own* wavelength-dependent PSF, so the synthetic blur exhibits
     genuine longitudinal-chromatic-aberration color fringing, not just achromatic blur.
  3. Run the *untrained* (random-init) `AdvancedUnfoldedOpticsDeblurNet` forward once,
     via `forward_from_camera` -- camera metadata straight in, no manual kernel
     construction -- to show the pipeline executes end-to-end on real data.
  4. Quickly fit the network's learnable parameters (per-channel mu_k and the proximal
     denoisers) against this single image pair with a few hundred gradient steps -- a
     per-image "does gradient descent through the unfolded physics actually converge"
     sanity check, not a claim of a generally-trained model.
  5. Visualize: ground truth | blurry/noisy input | untrained output | fitted output,
     with PSNR annotated, saved to outputs/deblur_demo.png.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.transform import resize

from unfolded_optics_deblur import InvertibleISP
from advanced_optics_kernel_engine import AdvancedOpticsKernelEngine, AdvancedUnfoldedOpticsDeblurNet

torch.manual_seed(0)
np.random.seed(0)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")
isp = InvertibleISP()

# ---------------------------------------------------------------------------
# 1. Real photograph
# ---------------------------------------------------------------------------
SIZE = 160
img = img_as_float(data.astronaut())                      # real photo, HxWx3, [0,1]
img = resize(img, (SIZE, SIZE), anti_aliasing=True)
sharp = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)   # (1,3,SIZE,SIZE), sRGB

# ---------------------------------------------------------------------------
# 2. Physically-derived diffraction + dispersion kernel + synthetic blurry/noisy
#    observation. The SAME `AdvancedOpticsKernelEngine` config is reused below to
#    build the model's internal kernel from metadata (`forward_from_camera`), so the
#    synthesis physics and the restoration physics are exactly consistent.
# ---------------------------------------------------------------------------
focal_length_m = torch.tensor([0.085])       # 85 mm portrait lens
f_number = torch.tensor([2.8])               # moderate defocus
# NOTE: a real sensor's native pixel pitch (~4-8 um) assumes a full-resolution frame
# (thousands of pixels wide). Our test image is a 160x160 thumbnail, so the pitch that
# is physically consistent with THIS crop is the sensor's *effective* per-pixel
# footprint at this resolution (i.e. pitch scales up as resolution scales down) -- not
# a literal native-sensor spec. Using the native pitch here would (correctly, per the
# same optics equations) predict a multi-hundred-pixel blur disc, which is simply too
# large to resolve on a 160px crop.
pixel_pitch_m = torch.tensor([8.0e-5])       # effective per-pixel footprint of this crop
focus_distance_m = torch.tensor([1.2])       # focused at 1.2 m
subject_distance_m = torch.tensor([2.0])     # background ~0.8 m behind focus plane

KSIZE = 25
DISPERSION_STRENGTH = 0.15   # 0 = achromatic lens, 1 = uncorrected single glass element
KERNEL_ENGINE_KWARGS = dict(dispersion_strength=DISPERSION_STRENGTH, combine_mode="fourier")

kernel_engine = AdvancedOpticsKernelEngine(**KERNEL_ENGINE_KWARGS)
kernel = kernel_engine(
    focal_length_m, f_number, pixel_pitch_m, focus_distance_m, subject_distance_m, ksize=KSIZE
)  # (1,3,K,K) -- one diffraction+dispersion PSF per RGB channel

centroid_y, centroid_x = torch.meshgrid(
    torch.arange(KSIZE, dtype=torch.float32) - (KSIZE - 1) / 2.0,
    torch.arange(KSIZE, dtype=torch.float32) - (KSIZE - 1) / 2.0,
    indexing="ij",
)
centroid_r = torch.sqrt(centroid_y.pow(2) + centroid_x.pow(2))
per_channel_radius = (kernel[0] * centroid_r.unsqueeze(0)).sum(dim=(-2, -1))
print(f"Per-channel (R,G,B) energy-weighted blur radius: "
      f"{[round(v, 3) for v in per_channel_radius.tolist()]} px  "
      f"(dispersion_strength={DISPERSION_STRENGTH})")

sharp_linear = isp.to_linear(sharp)
pad = KSIZE // 2
sharp_padded = F.pad(sharp_linear, (pad, pad, pad, pad), mode="reflect")
depthwise_kernel = kernel[0].unsqueeze(1)                   # (3,1,K,K) for groups=3 conv:
                                                              # each RGB channel convolved
                                                              # with its OWN wavelength's PSF.
blurry_linear = F.conv2d(sharp_padded, depthwise_kernel, groups=3)

SIGMA_VAL = 0.012
noise = SIGMA_VAL * torch.randn_like(blurry_linear)
blurry_linear_noisy = (blurry_linear + noise).clamp(0.0, 1.0)
blurry_srgb = isp.to_srgb(blurry_linear_noisy)

sigma_map = torch.full((1, 1, SIZE, SIZE), SIGMA_VAL)

# ---------------------------------------------------------------------------
# 3. Untrained (random-init) forward pass -- camera metadata straight in via
#    `forward_from_camera`; the model builds its own dispersion-aware kernel
#    internally (using the identical engine config), no manual kernel plumbing.
# ---------------------------------------------------------------------------
model = AdvancedUnfoldedOpticsDeblurNet(
    num_stages=6, base_ch=16, kernel_engine_kwargs=KERNEL_ENGINE_KWARGS
).to(device)
model.eval()
with torch.no_grad():
    restored_untrained = model.forward_from_camera(
        blurry_srgb, sigma_map, focal_length_m, f_number, pixel_pitch_m,
        focus_distance_m, subject_distance_m, ksize=KSIZE,
    )

# ---------------------------------------------------------------------------
# 4. Quick per-image fit (sanity-check that gradients through the unfolded
#    physics + Lipschitz denoisers actually let the network converge)
# ---------------------------------------------------------------------------
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
N_ITERS = 300

t0 = time.time()
for it in range(N_ITERS):
    optimizer.zero_grad()
    restored = model.forward_from_camera(
        blurry_srgb, sigma_map, focal_length_m, f_number, pixel_pitch_m,
        focus_distance_m, subject_distance_m, ksize=KSIZE,
    )
    loss = F.mse_loss(restored, sharp)
    loss.backward()
    optimizer.step()
    if it % 50 == 0 or it == N_ITERS - 1:
        print(f"iter {it:4d}/{N_ITERS}  mse={loss.item():.6f}")
print(f"Fit time: {time.time()-t0:.1f}s")

model.eval()
with torch.no_grad():
    restored_fitted = model.forward_from_camera(
        blurry_srgb, sigma_map, focal_length_m, f_number, pixel_pitch_m,
        focus_distance_m, subject_distance_m, ksize=KSIZE,
    ).clamp(0.0, 1.0)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


psnr_blurry = psnr(blurry_srgb, sharp)
psnr_untrained = psnr(restored_untrained, sharp)
psnr_fitted = psnr(restored_fitted, sharp)
print(f"PSNR  blurry input:      {psnr_blurry:.2f} dB")
print(f"PSNR  untrained output:  {psnr_untrained:.2f} dB")
print(f"PSNR  fitted output:     {psnr_fitted:.2f} dB")

# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------
def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()


fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
panels = [
    (to_np(sharp), "Ground truth (real photo)"),
    (to_np(blurry_srgb), f"Blurry + noisy input\nPSNR {psnr_blurry:.2f} dB"),
    (to_np(restored_untrained), f"Network output\n(random init)\nPSNR {psnr_untrained:.2f} dB"),
    (to_np(restored_fitted), f"Network output\n(after {N_ITERS} fit steps)\nPSNR {psnr_fitted:.2f} dB"),
]
for ax, (im, title) in zip(axes, panels):
    ax.imshow(im)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

fig.suptitle(
    "AdvancedUnfoldedOpticsDeblurNet on a real photograph (85mm f/2.8 synthetic "
    "defocus + diffraction + chromatic dispersion, per-channel R/G/B radius "
    f"{[round(v, 1) for v in per_channel_radius.tolist()]}px)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT_DIR, "deblur_demo.png")
fig.savefig(out_path, dpi=140)
print(f"Saved visualization to {out_path}")
