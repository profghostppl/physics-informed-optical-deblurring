# Physics-Informed Optical Deblurring

A non-generative neural architecture that reverses camera-lens defocus blur using
real optical physics and known (or EXIF-derived) camera hardware parameters —
focal length, f-number, sensor pixel pitch, focus distance, and subject distance.
The optics model covers geometric defocus, diffraction-limited blur (Airy disk),
and per-RGB-channel chromatic dispersion, not just a single achromatic blur circle.

**Core principle: recover, don't hallucinate.** This is not a diffusion model or an
unconstrained GAN. It never invents plausible-looking detail that wasn't in the
original signal. Every stage either solves a closed-form physical equation or runs
through a network mathematically constrained to be non-expansive (it can smooth
and denoise, but provably cannot fabricate new structure). A "pure physics" mode
lets you bypass the learned component entirely and see the literal mathematical
inverse of the blur, with nothing learned touching a pixel.

## Why this matters

Standard deconvolution ("unblurring") is an *ill-posed inverse problem*: a blur
kernel destroys certain spatial frequencies outright (multiplies them by
approximately zero), and that information is gone — not hidden, erased. No amount
of cleverness recovers a frequency component that was truly zeroed out. Most
"AI photo enhancers" hide this fact by using a generative model to *invent*
plausible-looking detail in place of what was lost, which produces convincing but
fabricated results — dangerous for anything where authenticity matters (forensics,
personal photos, historical restoration).

This project takes the opposite approach: solve the *exact* physically-justified
inverse where the math allows it (via a closed-form Wiener deconvolution derived
from the real optics), and apply only a conservative, provably-bounded cleanup step
elsewhere. The result won't be perfectly sharp — but everything in the output is
either the literal signal or a bounded denoising of it, never an invention.

## Architecture

The restoration problem is posed as classical energy minimization

```
min_x  (1/2) || y - K x ||^2  +  lambda * R(x)
```

and solved via **Half-Quadratic Splitting (HQS)**, unrolled into **K = 6 alternating
stages** ("deep unfolding" — the optimization structure is explicit in the network,
not learned from scratch):

1. **Data-fidelity step — closed-form, no learned pixels.**
   Solved analytically in the 2D Fourier domain via Wiener deconvolution:

   ```
   X^(k+1/2) = [ conj(K_fft) * Y_fft + mu_k * Z_fft ] / [ |K_fft|^2 + mu_k ]
   ```

   `mu_k` is learned in log-space (`mu_k = exp(alpha_k)`), which makes it strictly
   positive by construction — no division-by-zero or negative poles are possible,
   regardless of what value training finds. Implemented with `torch.fft.rfft2` /
   `irfft2` for real-valued efficiency. Reflect-padding plus a raised-cosine border
   taper suppress the ringing that naive zero/circular padding would otherwise
   inject from the image boundary (the DFT assumes periodic boundaries; real
   photos aren't periodic).

2. **Prior step — a Lipschitz-constrained learned denoiser.**
   Every convolution is wrapped in spectral normalization
   (`torch.nn.utils.parametrizations.spectral_norm`), and the network is a pure
   feed-forward composition — no additive residual connections inside blocks
   (only concatenation-based U-Net skips, which stay linear-safe under spectral
   norm). By sub-multiplicativity of Lipschitz constants, the whole network is
   therefore non-expansive. Under Monotone Operator Theory, plugging a
   non-expansive denoiser into this fixed-point iteration gives a stable,
   convergent scheme — not an unconstrained generator free to invent detail.
   A single global residual (`x - r(x)`, DnCNN-style) is applied once at the very
   end, conditioned on a per-pixel noise-level map so it adapts to sensor noise.

3. **Invertible ISP.** All arithmetic happens in *linear* radiometric space, via a
   differentiable sRGB↔linear mapping (IEC 61966-2-1 piecewise EOTF/OETF) — the
   linear convolution model `y = Kx + n` is only physically valid there; sRGB's
   gamma encoding does not sum linearly.

4. **Real optics, not arbitrary kernels.** The defocus blur-circle diameter is
   derived from the thin-lens equation:

   ```
   b(d) = f^2/N * |d - d0| / (d * (d0 - f))
   ```

   (`f` = focal length, `N` = f-number, `d0` = focus distance, `d` = subject
   distance), then converted to a pixel kernel radius via the sensor's pixel pitch.
   The kernel itself is a differentiable soft-edged disk (sigmoid-relaxed pillbox),
   so gradients can flow back through it to the physical parameters that produced it.

   `unfolded_optics_deblur.py` implements this base architecture
   (`UnfoldedOpticsDeblurNet`, single achromatic kernel per HQS stage). Every script
   in this repository actually runs the **upgraded** version below by default.

## Advanced optics engine

`advanced_optics_kernel_engine.py` upgrades kernel generation beyond the single
achromatic geometric pillbox above — it reuses `unfolded_optics_deblur.py`'s own
CoC derivation rather than duplicating it, and adds:

1. **Diffraction-limited PSF (`AdvancedOpticsKernelEngine`).** At narrow apertures
   or small pixel pitches, geometric optics alone under-predicts blur — wave optics
   (diffraction) contributes a real, non-negligible Airy disk. The engine convolves
   the geometric defocus pillbox with a true Airy diffraction pattern in the Fourier
   domain:

   ```
   I(r) = [ 2*J1(v) / v ]^2,   v = 3.8317 * r / r0,   r0 = 1.22 * lambda * N
   ```

   `torch.special.bessel_j1` has no autograd support in current PyTorch (verified:
   `.backward()` raises), so `J1` is evaluated via the Numerical Recipes
   rational/asymptotic polynomial approximation instead — fully differentiable,
   accurate to ~1e-8.

2. **Per-channel chromatic dispersion.** A two-term Cauchy dispersion relation
   `n(lambda) = A + B/lambda_um^2` gives each RGB wavelength its own effective focal
   length (fed back through the *same* CoC formula above), so red/green/blue get
   genuinely different defocus radii — a real 3-channel kernel `(B,3,Hk,Wk)`, not one
   kernel repeated three times. A `dispersion_strength` parameter (0 = achromatic
   lens, 1 = uncorrected single glass element) controls how much of the theoretical
   single-element dispersion survives a real compound lens's correction.

3. **Depth-guided spatially variant deconvolution** (`DepthToKernelField` +
   `OverlapAddSpatialDeconv`). A monocular relative depth map is soft-binned into a
   handful of representative-distance kernels; the image is deconvolved patch-wise
   with a per-patch blend of those kernels, then reassembled with Hann-window
   Overlap-Add, renormalized by the folded window energy so reconstruction is exact
   (no visible seams) regardless of window shape.

4. **Kernel-adaptive edge taper** (`DifferentiableEdgeTaper`). Before any `rfft2`
   call, the image is blended with its own circular self-blur, weighted by a border
   ramp whose *shape* is derived from the kernel's own cumulative energy profile
   (wide/soft kernels taper over a wide/soft border; narrow/sharp kernels taper
   sharply) — suppressing the Gibbs-ringing a hard image boundary would otherwise
   inject into FFT-based deconvolution.

5. **OTF zero-crossing safeguard** (`StabilizedMultiChannelWienerDeconv`). The
   Wiener regularizer is parametrized as `mu_k = softplus(alpha_k) + eps`, which
   floors the denominator `|K_fft|^2 + mu_k` strictly above zero *even exactly at* a
   diffraction OTF's Bessel null (`|K_fft| -> 0`) — bounding the noise-amplification
   gain everywhere in the spectrum, not just where the kernel happens to be
   well-conditioned. (The base model's data step instead parametrizes `mu_k =
   exp(alpha_k)`, which is also always positive but has no explicit floor.)

`AdvancedUnfoldedOpticsDeblurNet` (same file) is a drop-in upgrade of
`UnfoldedOpticsDeblurNet` — identical `forward(y_srgb, kernel, sigma)` interface —
that swaps in `StabilizedMultiChannelWienerDeconv` for every HQS stage's data step
and adds two convenience entry points:

- `forward_from_camera(y_srgb, sigma, f, N, pitch, d0, d)` — camera metadata straight
  in, kernel built internally.
- `forward_full_frame(y_srgb, sigma, depth_map, f, N, pitch, d0, d_min, d_max)` — runs
  every HQS stage patch-wise across a full image via the depth-guided Overlap-Add
  path above.

`UnfoldedOpticsDeblurNet` and the plain achromatic pillbox kernel remain available
in `unfolded_optics_deblur.py` as the base architecture the advanced engine builds
on (and as a smaller, faster path when diffraction/dispersion aren't needed).

## Real-world camera metadata pipeline

- **EXIF extraction** (`exif_utils.py`) pulls focal length and f-number directly
  from any photo's EXIF (reliable on nearly all camera JPEGs), and derives pixel
  pitch from `FocalPlaneXResolution`/`FocalPlaneResolutionUnit` when present —
  verified against real hardware (a Fujifilm X-S10's EXIF-derived pitch matched
  its published sensor spec almost exactly).
- **Sensor database fallback** (`camera_sensor_db.py`) covers cameras that omit
  that EXIF field entirely (confirmed on several Nikon DSLR bodies) with a
  two-tier lookup: a small table of standardized sensor *formats*
  (full-frame, APS-C variants, Micro Four Thirds, 1-inch, common smartphone
  formats) plus a `(Make, Model) → format` map that's cheap to extend. See
  [`SENSOR_REFERENCE.md`](SENSOR_REFERENCE.md) for the full human-readable table
  and its honest caveats (mirror-vs-mirrorless is *not* a physical variable here;
  smartphone sensor sizes are approximate and computational photography weakens
  this project's core single-exposure physical model for phone photos).
- **35mm-equivalent crop-factor fallback** — for cameras that write neither
  `FocalPlaneXResolution` nor match an entry in the sensor database, `exif_utils.py`
  falls back once more to the standard `FocalLengthIn35mmFilm` tag (present on
  nearly any camera with an auto/program exposure mode): `36mm / (35mm-equivalent
  focal length / actual focal length)` gives sensor width with no per-model lookup
  needed. Less precise than the database (both source fields are typically
  camera-rounded to the nearest mm) but per-photo and universally available.
- **What's honestly NOT recoverable from EXIF**: focus distance and subject
  distance have no reliable standard EXIF field across vendors — documented
  clearly in `exif_utils.py` rather than silently guessed. The app exposes these
  as user-adjustable sliders instead of pretending to auto-derive them.

## Interactive app

`app.py` (Streamlit): upload a photo, drag a resizable box over just the region you
want restored (a face, a sign, whatever), and run reconstruction on **only that
crop** — not the whole photo. This is a deliberate design choice, not just a speed
optimization:

- **Speed** — FFT deconvolution and the CNN both scale with pixel count; a small
  region is dramatically cheaper than a full-resolution photo.
- **Physical correctness** — the Wiener step assumes one shift-invariant blur
  kernel for the whole processed area, which is only really true if every pixel is
  at roughly the same depth. A whole photo mixes foreground/subject/background at
  different distances; a tight crop around one subject is much closer to the
  single-depth assumption the model relies on.

Other app features: EXIF auto-fill for optical parameters, a "pure physics mode"
toggle (skip the learned denoiser entirely and see the raw closed-form
reconstruction), auto noise-level estimation (a classical, non-learned estimator —
Immerkaer 1996 — computed correctly in linear-light space), and both a
region-only and a full-photo-with-region-restored download.

The app runs `AdvancedUnfoldedOpticsDeblurNet` (diffraction + per-channel chromatic
dispersion) by default. If `outputs/checkpoint.pt` was trained on the older,
single-achromatic-kernel architecture, the app loads whichever tensors still match
(the learned denoiser) and randomly re-initializes the rest (the optics data step)
rather than discarding the checkpoint outright — a sidebar note explains when this
happens.

## Project structure

```
unfolded_optics_deblur.py     Base architecture: InvertibleISP, AnalyticalWienerDeconv,
                             LipschitzProximalDenoiser, UnfoldedOpticsDeblurNet.
                             Includes a self-contained __main__ shape/gradient test.
advanced_optics_kernel_engine.py  Diffraction + chromatic-dispersion optics engine:
                             AdvancedOpticsKernelEngine, DepthToKernelField,
                             DifferentiableEdgeTaper, StabilizedMultiChannelWienerDeconv,
                             OverlapAddSpatialDeconv, AdvancedUnfoldedOpticsDeblurNet.
                             Includes a self-contained __main__ verification block.
training_optics_utils.py      Exact thin-lens CoC inversion used by the training
                             scripts to sample physically self-consistent camera
                             metadata for a target blur radius (see "Training
                             methodology" below).
exif_utils.py                Real-EXIF camera metadata extraction.
camera_sensor_db.py           Sensor-format fallback database for incomplete EXIF.
noise_estimation.py           Classical (non-learned) noise-level estimator.
dataset_sync.py                Cumulative, deduplicated real-photo dataset archive
                             with a stable, hash-derived train/validation split.
train_on_dataset.py            Training on public sample photographs (scikit-image),
                             for a quick end-to-end demonstration with no personal data.
train_on_real_photos.py        Training on a real, cumulative, EXIF-tagged photo
                             archive, with camera parameters bootstrap-resampled
                             from the archive's own real EXIF statistics.
demo_real_image.py             Single-image synthetic-blur sanity demo.
app.py                        Interactive Streamlit interface (see above).
run_all.py                    Runs the self-test, demo, and training stages in order.
SENSOR_REFERENCE.md            Human-readable camera sensor reference.
requirements.txt              Python dependencies.
```

## Installation

```bash
pip install -r requirements.txt
```

PyTorch will resolve a CPU build by default; for an NVIDIA GPU, install the CUDA
build first (see comments in `requirements.txt`).

## Usage

```bash
# Base architecture self-test (shapes, gradient flow) -- no data needed
python unfolded_optics_deblur.py

# Advanced optics engine self-test (diffraction, chromatic dispersion, depth-guided
# Overlap-Add, OTF zero-crossing safeguard, full AdvancedUnfoldedOpticsDeblurNet
# integration) -- no data needed
python advanced_optics_kernel_engine.py

# Single real-photo demo: synthesize diffraction + chromatic-dispersion blur with the
# project's own optics model, then fit the network to it (a per-image sanity check,
# not a trained model)
python demo_real_image.py

# Train on public sample photographs (no personal data required)
python train_on_dataset.py

# Train on your own real, EXIF-tagged photographs (see dataset_sync.py to point
# it at your own photo folder)
python train_on_real_photos.py

# Run the interactive app
streamlit run app.py

# Run everything above in sequence
python run_all.py
```

## Training methodology

Rather than sampling synthetic blur kernels from arbitrary uniform ranges,
`train_on_real_photos.py` **bootstrap-resamples (focal length, f-number, pixel
pitch) triples from a camera's own real EXIF statistics**, so the training
distribution reflects that camera/lens's actual joint behavior (e.g. its real
f-number range at each focal length it's actually shot at) rather than an
independently-sampled synthetic grid. Focus/subject distance, which EXIF can't
provide, are still randomly sampled — an explicit, documented approximation.

The dataset archive (`dataset_sync.py`) deduplicates by content hash (not
filename — camera frame counters can repeat) and assigns each photo's
train/validation split **once, deterministically, from its hash**, so held-out
validation stays a stable comparison point as more photos are added over time,
rather than reshuffling on every run.

Both training scripts sample a *target* geometric blur radius uniformly and
*solve* the thin-lens CoC equation for the subject distance that reproduces it
exactly (`training_optics_utils.py`), rather than sampling a free defocus offset
and clamping the resulting radius after the fact. Measured on the old
sample-then-clamp approach: without a clamp, 31% of samples exceeded the kernel
window (max observed radius ~245px), and the clamp step discarded the subject
distance that had produced the pre-clamp radius — harmless when only a scalar
radius fed a single achromatic kernel, but no longer harmless now that
`AdvancedOpticsKernelEngine` uses f-number and pixel pitch independently for
diffraction and focal length independently for dispersion, so a physically
consistent (f, N, pitch, d0, d) tuple actually matters.

### A real bug this workflow caught

Early real-photo testing surfaced a genuine failure mode: on night/low-light
photos, the learned denoiser produced visible darkening and blotchy artifacts.
Root-caused by isolating each pipeline stage: the noise level was being estimated
on raw sRGB pixel values but fed to the model as if already in linear-light space
— for dark images, sRGB's gamma curve compresses shadows heavily, so this
mismatch was measured at up to **6.2x too large** a noise estimate on a real dark
photo. Fixed by estimating noise after linearizing the crop. The deeper
contributor — the denoiser had rarely seen genuinely dark training patches — was
addressed by curating a diverse night-photo subset into the training archive.
Result, measured on held-out night-photo patches specifically (not just an
overall average): held-out night PSNR improved by **+2.5 dB**, essentially
matching the improvement seen on normally-lit patches, closing what had been a
real generalization gap.

## Known limitations

- **Focus and subject distance are not recoverable from EXIF** in general (see
  `exif_utils.py`); the app requires manual/estimated input for these.
- **Smartphone photos are a weaker fit for this architecture.** Nearly all phone
  photos are the product of multi-frame computational fusion (HDR stacking, Night
  Mode) before the JPEG is saved, which the single-exposure `y = Kx + n` model
  this project assumes does not describe as well as it does a single DSLR/
  mirrorless exposure.
- **Shift-invariant blur only, in the app.** Each `forward()` call assumes one blur
  kernel for the whole processed region — correct for a single depth plane, which is
  why the app restricts processing to a user-selected crop rather than a whole
  photo. `AdvancedUnfoldedOpticsDeblurNet.forward_full_frame` (depth-guided,
  patch-wise Overlap-Add) removes this restriction given a monocular depth map, but
  is not yet wired into the Streamlit UI.
- **This is a research/demonstration-scale project**, trained on a personally
  curated photo archive (not included in this repository — see below), not a
  large public benchmark dataset.

## A note on what's *not* in this repository

The training archive (`dataset/sharp/`), its manifest, and the resulting trained
checkpoint are excluded from version control. That archive contains real personal
and event photographs of identifiable people who did not consent to public
redistribution. The architecture, training code, and methodology are fully public
and reproducible against your own photos; the private data they were exercised on
is not.

## License

MIT — see [LICENSE](LICENSE).
