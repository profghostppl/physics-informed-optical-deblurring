# Physics-Informed Optical Deblurring

A non-generative neural architecture that reverses camera-lens defocus blur using
real optical physics and known (or EXIF-derived) camera hardware parameters —
focal length, f-number, sensor pixel pitch, focus distance, and subject distance.

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

## Project structure

```
unfolded_optics_deblur.py   Core architecture: InvertibleISP, AnalyticalWienerDeconv,
                             LipschitzProximalDenoiser, UnfoldedOpticsDeblurNet.
                             Includes a self-contained __main__ shape/gradient test.
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
# Architecture self-test (shapes, gradient flow) -- no data needed
python unfolded_optics_deblur.py

# Single real-photo demo: synthesize blur with the project's own optics model,
# then fit the network to it (a per-image sanity check, not a trained model)
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
- **Shift-invariant blur only.** The Wiener step assumes one blur kernel for the
  whole processed region — correct for a single depth plane, which is why the app
  restricts processing to a user-selected crop rather than a whole photo.
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
