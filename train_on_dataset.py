"""
Multi-image dataset training for AdvancedUnfoldedOpticsDeblurNet (replaces the earlier
single-image overfit demo with a proper train/held-out-validation split).

Since this environment has no internet-fetched dataset available, the "dataset" is
built from scikit-image's bundled real, public-domain photographs, expanded into a
large, diverse training distribution via:

  * random-crop patches (many distinct crops per source photo)
  * random horizontal/vertical flip augmentation
  * a *different randomly sampled optical configuration per patch* -- focal length,
    f-number, pixel pitch and focus distance are drawn from realistic ranges, and the
    subject distance is *solved for* (via `training_optics_utils`) so the resulting
    geometric defocus radius lands where intended, before the whole (f, N, pitch, d0,
    d) tuple is passed through `AdvancedOpticsKernelEngine` -- the same diffraction +
    chromatic-dispersion optical model the architecture itself uses -- so every
    training patch has its own physically-derived, per-RGB-channel defocus+diffraction
    kernel and noise level (heteroscedastic sigma)
  * a train/validation split at the IMAGE level: validation photographs (a cat photo
    and a retinal fundus photo) are held out completely and never seen during
    training, so validation PSNR measures generalization to unseen scene content and
    unseen (but in-distribution) blur/noise settings, not memorization of one image.

This is a demonstration-scale run (CPU-only, a handful of source photographs, a few
thousand patches) -- not a claim of a production-grade model trained on a large photo
corpus -- but it is trained the *proper* way: batches, an optimizer schedule, a loss
curve, and held-out evaluation, rather than gradient descent on a single image.
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
from training_optics_utils import sample_subject_distance_for_radius

torch.manual_seed(0)
np.random.seed(0)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")
isp = InvertibleISP()
KERNEL_ENGINE_KWARGS = dict(dispersion_strength=0.15, combine_mode="fourier")
kernel_engine = AdvancedOpticsKernelEngine(**KERNEL_ENGINE_KWARGS)

PATCH = 96
KSIZE = 25
BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Dataset: real photographs, split at the image level (train vs. held-out val)
# ---------------------------------------------------------------------------
def load_rgb(fn) -> torch.Tensor:
    img = img_as_float(fn())
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    # Upscale small images so PATCH-sized crops are meaningful.
    h, w = img.shape[:2]
    scale = max(1.0, (PATCH * 2) / min(h, w))
    if scale > 1.0:
        img = resize(img, (int(h * scale), int(w * scale)), anti_aliasing=True)
    return torch.from_numpy(img).float().permute(2, 0, 1)  # (3,H,W) in [0,1]


TRAIN_SOURCES = [load_rgb(f) for f in (
    data.astronaut, data.coffee, data.rocket, data.hubble_deep_field,
    data.immunohistochemistry, data.colorwheel,
)]
VAL_SOURCES = [load_rgb(f) for f in (data.chelsea, data.retina)]  # held-out, unseen scenes

print("Train source images:", [tuple(t.shape) for t in TRAIN_SOURCES])
print("Val   source images:", [tuple(t.shape) for t in VAL_SOURCES])


def sample_camera_params(n: int) -> tuple[torch.Tensor, ...]:
    """Sample n random, physically-plausible camera configs whose GEOMETRIC defocus
    radius is drawn uniformly from [1.0, 9.5]px (the KSIZE=25 window's usable range),
    by solving for the subject distance that reproduces each target radius exactly
    (see `training_optics_utils` for why, and for the measured old-vs-new comparison).
    Returns the raw (focal_length_m, f_number, pixel_pitch_m, focus_distance_m,
    subject_distance_m) tuple `AdvancedOpticsKernelEngine` needs, not a pre-baked
    radius -- it derives the diffraction + chromatic-dispersion kernel from these
    directly.
    """
    focal_length_m = torch.empty(n).uniform_(0.035, 0.135)     # 35-135mm
    f_number = torch.empty(n).uniform_(1.4, 5.6)
    pixel_pitch_m = torch.empty(n).uniform_(6.0e-5, 1.0e-4)    # effective per-pixel footprint
    focus_distance_m = torch.empty(n).uniform_(0.8, 3.0)
    radius_px_target = torch.empty(n).uniform_(1.0, 9.5)
    want_background = torch.randint(0, 2, (n,)).bool()

    subject_distance_m = sample_subject_distance_for_radius(
        focal_length_m, f_number, pixel_pitch_m, focus_distance_m, radius_px_target, want_background
    )
    return focal_length_m, f_number, pixel_pitch_m, focus_distance_m, subject_distance_m


def apply_batched_psf(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve each batch sample's (3,H,W) image with its OWN per-channel (3,kh,kw)
    dispersion-aware kernel -- each RGB channel convolved with its own wavelength's PSF.
    """
    b, c, h, w = x.shape
    _, kc, kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    weight = kernel.reshape(b * kc, 1, kh, kw)            # (B*C,1,kh,kw)
    x_reshaped = x_pad.reshape(1, b * c, x_pad.shape[-2], x_pad.shape[-1])
    out = F.conv2d(x_reshaped, weight, groups=b * c)
    return out.reshape(b, c, h, w)


def sample_batch(sources, batch_size: int, generator: torch.Generator = None):
    patches = []
    for _ in range(batch_size):
        img = sources[torch.randint(len(sources), (1,), generator=generator).item()]
        _, h, w = img.shape
        top = torch.randint(0, h - PATCH + 1, (1,), generator=generator).item()
        left = torch.randint(0, w - PATCH + 1, (1,), generator=generator).item()
        patch = img[:, top:top + PATCH, left:left + PATCH]
        if torch.rand(1, generator=generator).item() < 0.5:
            patch = patch.flip(-1)
        if torch.rand(1, generator=generator).item() < 0.5:
            patch = patch.flip(-2)
        patches.append(patch)
    sharp = torch.stack(patches, dim=0)                                   # (B,3,PATCH,PATCH)

    focal_length_m, f_number, pixel_pitch_m, focus_distance_m, subject_distance_m = sample_camera_params(batch_size)
    kernel = kernel_engine(
        focal_length_m, f_number, pixel_pitch_m, focus_distance_m, subject_distance_m, ksize=KSIZE
    )  # (B,3,K,K)

    sharp_linear = isp.to_linear(sharp)
    blurry_linear = apply_batched_psf(sharp_linear, kernel)
    sigma_val = torch.empty(batch_size, 1, 1, 1).uniform_(0.005, 0.02)
    blurry_linear_noisy = (blurry_linear + sigma_val * torch.randn_like(blurry_linear)).clamp(0.0, 1.0)
    blurry_srgb = isp.to_srgb(blurry_linear_noisy)
    sigma_map = sigma_val.expand(batch_size, 1, PATCH, PATCH)

    return blurry_srgb, kernel, sigma_map, sharp


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


# ---------------------------------------------------------------------------
# Fixed held-out validation batch (sampled once, reused every epoch)
# ---------------------------------------------------------------------------
val_gen = torch.Generator().manual_seed(123)
VAL_BATCH_SIZE = 16
val_blurry, val_kernel, val_sigma, val_sharp = sample_batch(VAL_SOURCES, VAL_BATCH_SIZE, generator=val_gen)

# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------
model = AdvancedUnfoldedOpticsDeblurNet(
    num_stages=6, base_ch=16, kernel_engine_kwargs=KERNEL_ENGINE_KWARGS
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# ---------------------------------------------------------------------------
# Auto-calibrate iteration budget to a wall-clock target (CPU-only environment)
# ---------------------------------------------------------------------------
TARGET_SECONDS = 240
WARMUP_ITERS = 5

model.train()
t0 = time.time()
for _ in range(WARMUP_ITERS):
    blurry, kernel, sigma_map, sharp = sample_batch(TRAIN_SOURCES, BATCH_SIZE)
    optimizer.zero_grad()
    restored = model(blurry, kernel, sigma_map)
    loss = F.mse_loss(restored, sharp)
    loss.backward()
    optimizer.step()
warmup_time = time.time() - t0
iter_time = warmup_time / WARMUP_ITERS
TOTAL_ITERS = int(np.clip(TARGET_SECONDS / iter_time, 400, 4000))
VAL_EVERY = max(TOTAL_ITERS // 15, 10)
print(f"Calibration: {iter_time*1000:.1f} ms/iter -> running {TOTAL_ITERS} iterations "
      f"(~{TOTAL_ITERS*iter_time:.0f}s), validating every {VAL_EVERY} iters")

# re-init model/optimizer so the warmup steps don't bias the reported curve
torch.manual_seed(0)
model = AdvancedUnfoldedOpticsDeblurNet(
    num_stages=6, base_ch=16, kernel_engine_kwargs=KERNEL_ENGINE_KWARGS
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_ITERS)

train_losses: list[float] = []
val_psnrs: list[float] = []
val_iters: list[int] = []

t_start = time.time()
for it in range(TOTAL_ITERS):
    model.train()
    blurry, kernel, sigma_map, sharp = sample_batch(TRAIN_SOURCES, BATCH_SIZE)
    optimizer.zero_grad()
    restored = model(blurry, kernel, sigma_map)
    loss = F.mse_loss(restored, sharp)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())

    if it % VAL_EVERY == 0 or it == TOTAL_ITERS - 1:
        model.eval()
        with torch.no_grad():
            val_restored = model(val_blurry, val_kernel, val_sigma).clamp(0.0, 1.0)
        val_psnr = psnr(val_restored, val_sharp)
        val_psnrs.append(val_psnr)
        val_iters.append(it)
        elapsed = time.time() - t_start
        print(f"iter {it:5d}/{TOTAL_ITERS}  train_mse={loss.item():.5f}  "
              f"val_psnr={val_psnr:.2f}dB  elapsed={elapsed:.0f}s")

total_time = time.time() - t_start
print(f"Training time: {total_time:.1f}s for {TOTAL_ITERS} iterations")

# ---------------------------------------------------------------------------
# Baseline PSNR (blurry vs sharp) on the SAME held-out validation batch
# ---------------------------------------------------------------------------
baseline_psnr = psnr(val_blurry, val_sharp)
model.eval()
with torch.no_grad():
    val_restored_final = model(val_blurry, val_kernel, val_sigma).clamp(0.0, 1.0)
final_psnr = psnr(val_restored_final, val_sharp)
print(f"Held-out validation PSNR -- blurry input: {baseline_psnr:.2f} dB | "
      f"trained model: {final_psnr:.2f} dB")

ckpt_path = os.path.join(OUT_DIR, "checkpoint.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"Saved checkpoint to {ckpt_path}")

# ---------------------------------------------------------------------------
# Visualization 1: training curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(train_losses, linewidth=0.8)
axes[0].set_title("Training MSE (per iteration)")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("MSE")
axes[0].set_yscale("log")

axes[1].plot(val_iters, val_psnrs, marker="o", label="Trained model (held-out val)")
axes[1].axhline(baseline_psnr, color="gray", linestyle="--", label="Blurry input baseline")
axes[1].set_title("Held-out validation PSNR")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("PSNR (dB)")
axes[1].legend(fontsize=9)

fig.suptitle(f"AdvancedUnfoldedOpticsDeblurNet training on real-photo dataset "
             f"({len(TRAIN_SOURCES)} train images / {len(VAL_SOURCES)} held-out val images)")
fig.tight_layout(rect=[0, 0, 1, 0.94])
curves_path = os.path.join(OUT_DIR, "training_curves.png")
fig.savefig(curves_path, dpi=140)
print(f"Saved training curves to {curves_path}")

# ---------------------------------------------------------------------------
# Visualization 2: qualitative results on held-out (never-trained-on) images
# ---------------------------------------------------------------------------
def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().permute(1, 2, 0).clamp(0, 1).numpy()


N_SHOW = 4
fig2, axes2 = plt.subplots(N_SHOW, 3, figsize=(9, 3 * N_SHOW))
for row in range(N_SHOW):
    axes2[row, 0].imshow(to_np(val_sharp[row]))
    axes2[row, 1].imshow(to_np(val_blurry[row]))
    axes2[row, 2].imshow(to_np(val_restored_final[row]))
    if row == 0:
        axes2[row, 0].set_title("Ground truth\n(held out)")
        axes2[row, 1].set_title(f"Blurry input\n{psnr(val_blurry[row:row+1], val_sharp[row:row+1]):.1f} dB")
        axes2[row, 2].set_title(f"Restored\n{psnr(val_restored_final[row:row+1], val_sharp[row:row+1]):.1f} dB")
    else:
        axes2[row, 1].set_title(f"{psnr(val_blurry[row:row+1], val_sharp[row:row+1]):.1f} dB")
        axes2[row, 2].set_title(f"{psnr(val_restored_final[row:row+1], val_sharp[row:row+1]):.1f} dB")
    for ax in axes2[row]:
        ax.axis("off")

fig2.suptitle("Held-out validation samples (cat photo + retina photo -- never seen during training)")
fig2.tight_layout(rect=[0, 0, 1, 0.96])
qual_path = os.path.join(OUT_DIR, "validation_qualitative.png")
fig2.savefig(qual_path, dpi=140)
print(f"Saved qualitative validation panel to {qual_path}")
