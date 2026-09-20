"""
Real-image test/demo for UnfoldedOpticsDeblurNet.

There is no pretrained checkpoint (the model in unfolded_optics_deblur.py ships with
random init only), so a single random forward pass would not look "deblurred" -- it
would just prove the plumbing works. To actually *demonstrate* the physics-informed
mechanism on a real photograph we do the following, entirely with real image content:

  1. Take a real, public-domain test photograph (scikit-image's bundled "astronaut"
     sample -- a real photo of astronaut Eileen Collins, shipped with the library
     precisely for this kind of CV testing).
  2. Synthesize a physically realistic defocus-blurred + noisy observation from it,
     using our own `circle_of_confusion_diameter_m` -> pixel-radius -> pillbox-kernel
     pipeline (i.e. the exact same optics model the network assumes), plus additive
     sensor noise.
  3. Run the *untrained* (random-init) network forward once, to show the pipeline
     executes end-to-end on real data.
  4. Quickly fit the network's learnable parameters (mu_k and the proximal denoisers)
     against this single image pair with a few hundred gradient steps -- a per-image
     "does gradient descent through the unfolded physics actually converge" sanity
     check, not a claim of a generally-trained model.
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

from unfolded_optics_deblur import (
    InvertibleISP,
    UnfoldedOpticsDeblurNet,
    circle_of_confusion_diameter_m,
    defocus_diameter_to_pixel_radius,
    make_soft_pillbox_kernel,
)

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
# 2. Physically-derived defocus kernel + synthetic blurry/noisy observation
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

b_m = circle_of_confusion_diameter_m(focal_length_m, f_number, focus_distance_m, subject_distance_m)
radius_px = defocus_diameter_to_pixel_radius(b_m, pixel_pitch_m)
print(f"CoC diameter: {b_m.item()*1e3:.3f} mm  ->  kernel radius: {radius_px.item():.2f} px")

KSIZE = 25
kernel = make_soft_pillbox_kernel(radius_px, ksize=KSIZE, softness_px=0.6)   # (1,1,K,K)

sharp_linear = isp.to_linear(sharp)
pad = KSIZE // 2
sharp_padded = F.pad(sharp_linear, (pad, pad, pad, pad), mode="reflect")
depthwise_kernel = kernel.repeat(3, 1, 1, 1)                # (3,1,K,K) for groups=3 conv
blurry_linear = F.conv2d(sharp_padded, depthwise_kernel, groups=3)

SIGMA_VAL = 0.012
noise = SIGMA_VAL * torch.randn_like(blurry_linear)
blurry_linear_noisy = (blurry_linear + noise).clamp(0.0, 1.0)
blurry_srgb = isp.to_srgb(blurry_linear_noisy)

sigma_map = torch.full((1, 1, SIZE, SIZE), SIGMA_VAL)

# ---------------------------------------------------------------------------
# 3. Untrained (random-init) forward pass
# ---------------------------------------------------------------------------
model = UnfoldedOpticsDeblurNet(num_stages=6, base_ch=16).to(device)
model.eval()
with torch.no_grad():
    restored_untrained = model(blurry_srgb, kernel, sigma_map)

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
    restored = model(blurry_srgb, kernel, sigma_map)
    loss = F.mse_loss(restored, sharp)
    loss.backward()
    optimizer.step()
    if it % 50 == 0 or it == N_ITERS - 1:
        print(f"iter {it:4d}/{N_ITERS}  mse={loss.item():.6f}")
print(f"Fit time: {time.time()-t0:.1f}s")

model.eval()
with torch.no_grad():
    restored_fitted = model(blurry_srgb, kernel, sigma_map).clamp(0.0, 1.0)


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
    "UnfoldedOpticsDeblurNet on a real photograph (85mm f/1.8 synthetic defocus, "
    f"kernel radius {radius_px.item():.1f}px)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT_DIR, "deblur_demo.png")
fig.savefig(out_path, dpi=140)
print(f"Saved visualization to {out_path}")
