"""
Train UnfoldedOpticsDeblurNet on the user's OWN real photographs (Fujifilm X-S10,
EXIF intact), instead of the 6 generic scikit-image stock photos used in
train_on_dataset.py.

What's genuinely different from train_on_dataset.py here:
  * Ground-truth sharp images are the user's real, high-resolution (6240x3512+)
    photographs, not small stock test images -- far more diverse real-world
    content (portraits, pets, landscapes, macro) at native camera resolution.
  * Synthetic blur kernels are no longer sampled from arbitrary uniform ranges for
    focal length / f-number / pixel pitch. Instead, every training patch draws its
    (focal_length_mm, f_number, pixel_pitch_m) triple by BOOTSTRAP RESAMPLING (with
    replacement) from the REAL EXIF values actually recorded across this camera's
    photos -- so the synthetic training distribution matches the real joint
    correlations of this camera/lens (e.g. this lens's f-number range at each focal
    length), not an independently-sampled synthetic grid.
  * The pixel pitch itself is independently cross-checked: EXIF's focal-plane-
    resolution-derived pitch (~3.76um) matches the X-S10's published APS-C sensor
    spec (23.5x15.6mm / 6240x4160) almost exactly, which is a good sanity check that
    exif_utils.py's physics is right, not just self-consistent.

What's still an approximation, honestly: focus distance (d0) and subject distance
(d) are NOT recoverable from this camera's EXIF (see exif_utils.py docstring), so
they are still randomly sampled per patch, same as train_on_dataset.py. Only the
optical parameters that genuinely determine aperture/focal-length behavior are now
grounded in real camera data.

Dataset sourcing: this script no longer reads the user's Desktop drop folder
directly (that folder's contents have already been replaced once, which would have
silently shrunk the dataset back down). Instead it calls dataset_sync.sync_from_source()
first, which copies any new photos from that folder into the permanent archive at
dataset/sharp/ (deduplicated by content hash) and assigns each photo a train/val
split ONCE, deterministically, from its hash -- so held-out validation stays a
stable comparison point across runs even as more photos get added over time. Every
run trains on the FULL cumulative archive, not just the most recent drop.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from unfolded_optics_deblur import (
    InvertibleISP,
    UnfoldedOpticsDeblurNet,
    circle_of_confusion_diameter_m,
    defocus_diameter_to_pixel_radius,
    make_soft_pillbox_kernel,
)
from exif_utils import extract_camera_metadata
from dataset_sync import sync_from_source, sync_files, ARCHIVE_DIR

# Hand-picked, genuinely dark/night photos from a wedding shoot (Nikon D60/D5100/
# D5300 across two ceremonies), pulled out of a much larger folder (~800 photos, most
# of them normally lit) specifically to give the denoiser real low-light examples --
# see the conversation this was diagnosed in: the CNN denoiser was found to darken
# and introduce blotchy artifacts on genuinely dark real photos because training data
# so far was almost entirely normally-lit. Picked to be diverse (spread across the
# darkness ranking, not a single burst of near-identical consecutive frames) rather
# than mechanically the single darkest N, which was dominated by repeated bursts.
NIGHT_SUBSET_DIR = Path(r"E:\Manish Vs Swati(Patna)")
NIGHT_SUBSET_FILES = [
    NIGHT_SUBSET_DIR / name for name in [
        "_DSC0507.JPG", "_DSC0383.JPG", "_DSC0459.JPG", "DSC_1652.JPG",
        "_DSC0434.JPG", "_DSC0454.JPG", "DSC_1518.JPG", "DSC_0170.JPG",
        "_DSC0448.JPG", "_DSC0399.JPG", "DSC_0342.JPG", "DSC_1588.JPG",
        "DSC_1630.JPG", "_DSC0516.JPG", "_DSC0336.JPG", "DSC_1672.JPG",
        "DSC_1679.JPG",
    ]
]

# A broader (not night-specific) subset from a second wedding shoot -- three more
# Nikon bodies (D300S, D7500, D5600), 993 photos total across 5 camera-roll folders.
# Rather than archive all 993 (heavily redundant consecutive-burst frames) or hand-pick
# individual files, this is an evenly-spaced sample within each folder: enough frames
# apart to represent genuinely different moments/scenes, roughly proportional to each
# folder's size, with the small D5600 folder taken in full since it's already small.
_MITTHU_BASE = Path(r"E:\Mitthu Marraige\New folder 20")
_MITTHU_PLAN = {
    "108D5600": None,       # None = take all (already small)
    "100D7500 01": 17,
    "101D7500": 13,
    "337D300S 04": 35,
    "337D300S": 17,
}


def _evenly_spaced_sample(folder: str, n) -> list:
    files = sorted((_MITTHU_BASE / folder).glob("*.JPG"))
    if n is None or n >= len(files):
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


MITTHU_SUBSET_FILES = [
    p for folder, n in _MITTHU_PLAN.items() for p in _evenly_spaced_sample(folder, n)
]

torch.manual_seed(0)
np.random.seed(0)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")
isp = InvertibleISP()

PATCH = 96
KSIZE = 25
BATCH_SIZE = 8
MAX_SOURCE_DIM = 1600  # cap in-memory resolution; still far larger than 96px patches


def load_photo(path: Path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_SOURCE_DIM / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    meta = extract_camera_metadata(str(path))
    return np.asarray(img), meta  # uint8 HxWx3


manifest = sync_from_source()
manifest = sync_files(NIGHT_SUBSET_FILES, tag="night_subset")
manifest = sync_files(MITTHU_SUBSET_FILES, tag="mitthu_marriage")

train_images, train_params = [], []
val_images, val_params, val_names, val_tags = [], [], [], []
night_train_count, night_val_count = 0, 0

for content_hash, entry in sorted(manifest.items(), key=lambda kv: kv[1]["archive_filename"]):
    path = ARCHIVE_DIR / entry["archive_filename"]
    arr, meta = load_photo(path)
    if meta.focal_length_mm is None or meta.f_number is None or meta.pixel_pitch_m is None:
        print(f"Skipping {entry['archive_filename']}: incomplete EXIF")
        continue
    params = (meta.focal_length_mm, meta.f_number, meta.pixel_pitch_m)
    is_night = entry.get("tag") == "night_subset"
    if entry["split"] == "val":
        val_images.append(arr)
        val_params.append(params)
        val_names.append(entry["archive_filename"])
        val_tags.append(entry.get("tag"))
        night_val_count += is_night
    else:
        train_images.append(arr)
        train_params.append(params)
        night_train_count += is_night

print(f"Train images: {len(train_images)} | Held-out validation images: {len(val_images)}")
print(f"Night subset: {night_train_count} in train, {night_val_count} in held-out val")
print("Held-out files:", val_names)

REAL_PARAM_POOL = np.array(train_params)  # (N, 3): focal_length_mm, f_number, pixel_pitch_m


def sample_real_camera_params(n: int) -> torch.Tensor:
    """Bootstrap-resample (f, N, pitch) triples from this camera's REAL EXIF values,
    preserving real joint correlations, then derive the resulting blur pixel radius.
    Focus/subject distance are NOT in EXIF (see module docstring) -- randomly sampled.
    """
    idx = np.random.randint(0, len(REAL_PARAM_POOL), size=n)
    sampled = REAL_PARAM_POOL[idx]  # (n, 3)
    focal_length_m = torch.tensor(sampled[:, 0] / 1000.0, dtype=torch.float32)
    f_number = torch.tensor(sampled[:, 1], dtype=torch.float32)
    pixel_pitch_m = torch.tensor(sampled[:, 2], dtype=torch.float32)

    focus_distance_m = torch.empty(n).uniform_(0.8, 3.0)
    delta = torch.empty(n).uniform_(0.4, 1.8) * (torch.randint(0, 2, (n,)) * 2 - 1)
    subject_distance_m = (focus_distance_m + delta).clamp_min(0.3)

    b_m = circle_of_confusion_diameter_m(focal_length_m, f_number, focus_distance_m, subject_distance_m)
    radius_px = defocus_diameter_to_pixel_radius(b_m, pixel_pitch_m)
    return radius_px.clamp(1.0, 9.5)  # keep within the KSIZE=25 window with margin


def apply_batched_psf(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    _, _, kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    weight = kernel.repeat_interleave(c, dim=0)
    x_reshaped = x_pad.reshape(1, b * c, x_pad.shape[-2], x_pad.shape[-1])
    out = F.conv2d(x_reshaped, weight, groups=b * c)
    return out.reshape(b, c, h, w)


def sample_batch(images: list, batch_size: int, generator: torch.Generator = None):
    patches = []
    source_indices = []
    for _ in range(batch_size):
        idx = torch.randint(len(images), (1,), generator=generator).item()
        source_indices.append(idx)
        arr = images[idx]
        h, w = arr.shape[:2]
        top = torch.randint(0, h - PATCH + 1, (1,), generator=generator).item()
        left = torch.randint(0, w - PATCH + 1, (1,), generator=generator).item()
        patch = torch.from_numpy(arr[top:top + PATCH, left:left + PATCH].copy()).permute(2, 0, 1).float() / 255.0
        if torch.rand(1, generator=generator).item() < 0.5:
            patch = patch.flip(-1)
        if torch.rand(1, generator=generator).item() < 0.5:
            patch = patch.flip(-2)
        patches.append(patch)
    sharp = torch.stack(patches, dim=0)

    radius_px = sample_real_camera_params(batch_size)
    kernel = make_soft_pillbox_kernel(radius_px, ksize=KSIZE, softness_px=0.6)

    sharp_linear = isp.to_linear(sharp)
    blurry_linear = apply_batched_psf(sharp_linear, kernel)
    sigma_val = torch.empty(batch_size, 1, 1, 1).uniform_(0.005, 0.02)
    blurry_linear_noisy = (blurry_linear + sigma_val * torch.randn_like(blurry_linear)).clamp(0.0, 1.0)
    blurry_srgb = isp.to_srgb(blurry_linear_noisy)
    sigma_map = sigma_val.expand(batch_size, 1, PATCH, PATCH)

    return blurry_srgb, kernel, sigma_map, sharp, source_indices


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


val_gen = torch.Generator().manual_seed(123)
VAL_BATCH_SIZE = 16
# Temporarily point the sampler at the held-out pool for building the fixed val batch.
_train_pool_backup = REAL_PARAM_POOL
REAL_PARAM_POOL = np.array(val_params)
val_blurry, val_kernel, val_sigma, val_sharp, val_source_idx = sample_batch(
    val_images, VAL_BATCH_SIZE, generator=val_gen
)
REAL_PARAM_POOL = _train_pool_backup
val_is_night = [val_tags[i] == "night_subset" for i in val_source_idx]
print(f"Fixed val batch: {sum(val_is_night)}/{VAL_BATCH_SIZE} patches are from night-tagged photos")

model = UnfoldedOpticsDeblurNet(num_stages=6, base_ch=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

TARGET_SECONDS = 360  # bumped from 300s: larger, more diverse dataset (76 vs 40 photos, 2 lenses)
WARMUP_ITERS = 5

model.train()
t0 = time.time()
for _ in range(WARMUP_ITERS):
    blurry, kernel, sigma_map, sharp, _ = sample_batch(train_images, BATCH_SIZE)
    optimizer.zero_grad()
    restored = model(blurry, kernel, sigma_map)
    loss = F.mse_loss(restored, sharp)
    loss.backward()
    optimizer.step()
iter_time = (time.time() - t0) / WARMUP_ITERS
TOTAL_ITERS = int(np.clip(TARGET_SECONDS / iter_time, 400, 4000))
VAL_EVERY = max(TOTAL_ITERS // 15, 10)
print(f"Calibration: {iter_time*1000:.1f} ms/iter -> running {TOTAL_ITERS} iterations "
      f"(~{TOTAL_ITERS*iter_time:.0f}s), validating every {VAL_EVERY} iters")

torch.manual_seed(0)
model = UnfoldedOpticsDeblurNet(num_stages=6, base_ch=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_ITERS)

train_losses, val_psnrs, val_iters = [], [], []
t_start = time.time()
for it in range(TOTAL_ITERS):
    model.train()
    blurry, kernel, sigma_map, sharp, _ = sample_batch(train_images, BATCH_SIZE)
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
        vp = psnr(val_restored, val_sharp)
        val_psnrs.append(vp)
        val_iters.append(it)
        print(f"iter {it:5d}/{TOTAL_ITERS}  train_mse={loss.item():.5f}  "
              f"val_psnr={vp:.2f}dB  elapsed={time.time()-t_start:.0f}s")

print(f"Training time: {time.time()-t_start:.1f}s for {TOTAL_ITERS} iterations")

baseline_psnr = psnr(val_blurry, val_sharp)
model.eval()
with torch.no_grad():
    val_restored_final = model(val_blurry, val_kernel, val_sigma).clamp(0.0, 1.0)
final_psnr = psnr(val_restored_final, val_sharp)
print(f"Held-out validation PSNR -- blurry input: {baseline_psnr:.2f} dB | trained model: {final_psnr:.2f} dB")

night_idx = [i for i, is_n in enumerate(val_is_night) if is_n]
other_idx = [i for i, is_n in enumerate(val_is_night) if not is_n]
if night_idx:
    night_baseline = psnr(val_blurry[night_idx], val_sharp[night_idx])
    night_final = psnr(val_restored_final[night_idx], val_sharp[night_idx])
    other_baseline = psnr(val_blurry[other_idx], val_sharp[other_idx])
    other_final = psnr(val_restored_final[other_idx], val_sharp[other_idx])
    print(f"  Night-tagged val patches ({len(night_idx)}):   blurry {night_baseline:.2f} dB -> trained {night_final:.2f} dB")
    print(f"  Other val patches ({len(other_idx)}):          blurry {other_baseline:.2f} dB -> trained {other_final:.2f} dB")

ckpt_path = os.path.join(OUT_DIR, "checkpoint.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"Saved checkpoint to {ckpt_path}")

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
fig.suptitle(f"Trained on {len(train_images)} real EXIF-tagged photos "
             f"(Fujifilm + Nikon) / {len(val_images)} held-out, "
             f"incl. {night_train_count + night_val_count} night-subset photos")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "training_curves_real.png"), dpi=140)
print("Saved training_curves_real.png")


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().permute(1, 2, 0).clamp(0, 1).numpy()


N_SHOW = min(6, VAL_BATCH_SIZE)
# Prioritize night-tagged patches first so the night-specific improvement is visible
# at a glance, rather than possibly diluted/absent among mostly-normal-lit patches.
show_order = night_idx[:N_SHOW] + [i for i in other_idx if i not in night_idx][: max(0, N_SHOW - len(night_idx))]
fig2, axes2 = plt.subplots(N_SHOW, 3, figsize=(9, 3 * N_SHOW))
for row, src_i in enumerate(show_order):
    axes2[row, 0].imshow(to_np(val_sharp[src_i]))
    axes2[row, 1].imshow(to_np(val_blurry[src_i]))
    axes2[row, 2].imshow(to_np(val_restored_final[src_i]))
    night_label = " [NIGHT]" if val_is_night[src_i] else ""
    axes2[row, 1].set_title(f"{psnr(val_blurry[src_i:src_i+1], val_sharp[src_i:src_i+1]):.1f} dB{night_label}")
    axes2[row, 2].set_title(f"{psnr(val_restored_final[src_i:src_i+1], val_sharp[src_i:src_i+1]):.1f} dB{night_label}")
    if row == 0:
        axes2[row, 0].set_title("Ground truth\n(held out)")
    for ax in axes2[row]:
        ax.axis("off")
fig2.suptitle(f"Held-out validation patches from: {', '.join(val_names)}")
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(os.path.join(OUT_DIR, "validation_qualitative_real.png"), dpi=140)
print("Saved validation_qualitative_real.png")
