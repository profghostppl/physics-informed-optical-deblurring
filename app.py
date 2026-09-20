"""
Streamlit interface: upload a photo, drag a rectangle over just the region you want
deblurred (a face, a license plate, whatever), tune/confirm the optical parameters
(auto-filled from EXIF where genuinely available), run the physics-informed
deblurring network on ONLY that region, and download the result.

Why crop instead of processing the whole photo:
  * Speed -- FFT deconvolution + the CNN prior both scale with pixel count, so a
    small selected region is dramatically cheaper than a full-resolution photo.
  * Physical correctness -- the Wiener deconvolution step assumes ONE shift-invariant
    blur kernel for the whole processed area. That assumption is only really true if
    every pixel is at roughly the same depth. A whole photo mixes foreground,
    subject, and background at different distances; a tight crop around one face or
    object is much closer to the single-depth assumption the model relies on.
  * Distribution match -- the network was trained on 96x96 patches (see
    train_on_dataset.py); a small selected crop stays closer to that training
    resolution than an arbitrary full-photo downscale would.

Run with:
    streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_cropper import st_cropper

from unfolded_optics_deblur import InvertibleISP, circle_of_confusion_diameter_m, defocus_diameter_to_pixel_radius
from advanced_optics_kernel_engine import AdvancedUnfoldedOpticsDeblurNet
from exif_utils import extract_camera_metadata
from noise_estimation import estimate_noise_sigma

ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "outputs" / "checkpoint.pt"
CROP_MAX_DIM = 320  # cap the SELECTED REGION's resolution for CPU-feasible processing

st.set_page_config(page_title="Optics-Aware Deblurring", layout="wide")

_isp = InvertibleISP()


def to_linear_np(srgb_np: np.ndarray) -> np.ndarray:
    """sRGB [0,1] HxWx3 numpy -> linear-light [0,1] HxWx3 numpy, via InvertibleISP."""
    t = torch.from_numpy(srgb_np).permute(2, 0, 1).unsqueeze(0)
    return _isp.to_linear(t).squeeze(0).permute(1, 2, 0).numpy()


@st.cache_resource
def load_model():
    model = AdvancedUnfoldedOpticsDeblurNet(
        num_stages=6, base_ch=16, kernel_engine_kwargs=dict(dispersion_strength=0.15, combine_mode="fourier")
    )
    loaded_checkpoint = False
    if CKPT_PATH.exists():
        try:
            state = torch.load(CKPT_PATH, map_location="cpu")
            own_state = model.state_dict()
            # A checkpoint saved by the previous single-achromatic-kernel architecture
            # has differently-shaped `data_steps.*` tensors (scalar mu vs. this model's
            # per-channel mu) -- keep whichever tensors still match exactly (the learned
            # denoiser, `priors.*`, is architecturally unchanged) and let the rest fall
            # back to random init, rather than discarding the whole checkpoint.
            compatible = {k: v for k, v in state.items() if k in own_state and own_state[k].shape == v.shape}
            model.load_state_dict(compatible, strict=False)
            loaded_checkpoint = True
            if len(compatible) < len(own_state):
                st.sidebar.info(
                    f"Checkpoint was trained on the previous architecture: loaded "
                    f"{len(compatible)}/{len(own_state)} matching tensors (the learned "
                    "denoiser); the upgraded optics data step was randomly re-"
                    "initialized. Re-run train_on_dataset.py for a fully trained checkpoint."
                )
        except Exception as e:
            st.sidebar.error(f"Checkpoint found but failed to load ({e}); using random init.")
    model.eval()
    return model, loaded_checkpoint


def cap_resolution(pil_img: Image.Image, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale for CPU-feasible processing; also return the scale factor applied.

    EXIF-derived pixel pitch describes the camera's NATIVE sensor resolution. If we
    hand the physics a downscaled crop without also inflating the pitch, the blur
    radius comes out wrong by the downscale factor (a "pixel" in the working crop
    now spans many native sensor pixels). Callers must divide the native pitch by
    this returned scale before computing the kernel radius.
    """
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.asarray(rgb).astype(np.float32) / 255.0, scale


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


model, ckpt_loaded = load_model()

st.title("Physics-Informed Optical Deblurring")
st.caption(
    "Deep-unfolded Half-Quadratic Splitting + closed-form Wiener deconvolution, "
    "conditioned on the camera's real optical parameters -- not a generative model."
)
if not ckpt_loaded:
    st.sidebar.warning(
        "No trained checkpoint found at outputs/checkpoint.pt -- running with "
        "randomly initialized denoiser weights. Only the physics-based Wiener step "
        "will behave meaningfully; run train_on_dataset.py first for a trained prior."
    )
else:
    st.sidebar.success("Loaded trained checkpoint from outputs/checkpoint.pt")

st.sidebar.header("1. Upload a photo")
uploaded = st.sidebar.file_uploader("Photo", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])

if uploaded is None:
    st.info("Upload a photo from the sidebar to begin.")
    st.stop()

raw_bytes = uploaded.getvalue()
pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
meta = extract_camera_metadata(io.BytesIO(raw_bytes))

st.sidebar.header("2. Camera / optics parameters")
if meta.has_exif:
    st.sidebar.caption(f"EXIF found -- camera: {meta.make or '?'} {meta.model or ''}")
    if meta.lens_model:
        st.sidebar.caption(f"Lens: {meta.lens_model}")
else:
    st.sidebar.caption("No EXIF found in this file (common for edited/re-saved images) -- enter values manually.")

focal_length_mm = st.sidebar.number_input(
    "Focal length f (mm)",
    min_value=1.0, max_value=1000.0,
    value=float(meta.focal_length_mm) if meta.focal_length_mm else 50.0,
    help="From EXIF FocalLength when available.",
)
f_number = st.sidebar.number_input(
    "f-number N",
    min_value=0.5, max_value=32.0,
    value=float(meta.f_number) if meta.f_number else 2.8,
    help="From EXIF FNumber when available.",
)

if meta.pixel_pitch_m and meta.pixel_pitch_source == "exif_focal_plane":
    st.sidebar.caption(
        f"Pixel pitch derived from EXIF focal-plane resolution: {meta.pixel_pitch_m * 1e6:.2f} um "
        "(this is the camera's NATIVE pitch; it will be auto-adjusted below if your "
        "selected region needs downscaling for processing)."
    )
    default_pitch_um = meta.pixel_pitch_m * 1e6
elif meta.pixel_pitch_m and meta.pixel_pitch_source == "known_sensor_fallback":
    st.sidebar.caption(
        f"This camera doesn't write focal-plane-resolution EXIF, so pixel pitch "
        f"({meta.pixel_pitch_m * 1e6:.2f} um) is from a published sensor-spec lookup "
        f"for {meta.make} {meta.model}, not a per-photo EXIF reading -- verify if precision matters."
    )
    default_pitch_um = meta.pixel_pitch_m * 1e6
else:
    st.sidebar.caption("Pixel pitch not in EXIF and camera model not recognized -- estimate, or leave the default.")
    default_pitch_um = 5.0
pixel_pitch_um = st.sidebar.number_input(
    "Sensor pixel pitch p (um)", min_value=0.5, max_value=200.0, value=float(default_pitch_um)
)

st.sidebar.markdown(
    "**Focus & subject distance:** standard EXIF has no reliable public field for "
    "these (see `exif_utils.py` docstring) -- please estimate."
)
if meta.subject_distance_m:
    st.sidebar.caption(
        f"EXIF reports a SubjectDistance of {meta.subject_distance_m:.2f} m, but this "
        "tag is low-confidence across camera vendors -- verify before trusting it."
    )
focus_distance_m = st.sidebar.slider("Focus distance d0 (m)", 0.2, 15.0, 1.5, 0.1)
subject_distance_m = st.sidebar.slider(
    "Subject distance d (m)", 0.2, 20.0,
    float(meta.subject_distance_m) if meta.subject_distance_m else 2.5, 0.1,
)

st.subheader("3. Select the region to deblur")
st.caption(
    "Drag the box over just the area you want restored (a face, a sign, etc.) -- only "
    "this region is processed, not the whole photo. Full-resolution detail from the "
    "original upload is preserved for the selection."
)
cropped_img, box = st_cropper(
    pil_img, realtime_update=True, box_color="#FF4B4B", aspect_ratio=None, return_type="both"
)
st.caption(
    f"Selected region: {box['width']}x{box['height']}px at ({box['left']}, {box['top']}) "
    f"of the {pil_img.width}x{pil_img.height}px original."
)

crop_working, crop_scale = cap_resolution(cropped_img, CROP_MAX_DIM)
if crop_scale < 1.0:
    st.caption(
        f"Selection downscaled {1/crop_scale:.2f}x (to {crop_working.shape[1]}x{crop_working.shape[0]}) "
        f"for CPU-feasible processing -- pick a smaller box to process at full detail."
    )
effective_pixel_pitch_um = pixel_pitch_um / crop_scale

st.sidebar.header("4. Noise level")
auto_sigma = estimate_noise_sigma(to_linear_np(crop_working))
st.sidebar.caption(
    f"Auto-estimated in linear-light space (Immerkaer 1996): {auto_sigma:.4f}. "
    "Estimating on the raw sRGB pixels instead would systematically overstate noise "
    "on dark/night shots (sRGB's gamma curve compresses shadows, so a given sRGB "
    "delta corresponds to a much smaller true linear-light delta there) -- that "
    "mismatch was previously causing visible darkening/artifacts on night photos."
)
sigma_val = st.sidebar.slider("Noise sigma (override if needed)", 0.0, 0.1, float(auto_sigma), 0.001)

st.sidebar.header("5. Reconstruction mode")
pure_physics_mode = st.sidebar.checkbox("Pure physics mode (no learned denoising)", value=False)
st.sidebar.caption(
    "Off (default): closed-form Wiener deconvolution + the trained, spectrally-"
    "constrained CNN denoiser (smooths noise the physics step can't resolve).\n\n"
    "On: skips the CNN entirely. Every stage still runs the same deterministic "
    "Wiener equation -- no learned network ever touches a pixel -- but frequencies "
    "the blur kernel actually destroyed stay noisy/ringy instead of being smoothed "
    "over. Nothing is invented either way; this toggle only controls whether the "
    "conservative cleanup step runs."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Selected region")
    st.image(crop_working, use_container_width=True, clamp=True)

run = st.sidebar.button("Run deblurring", type="primary")

with col2:
    st.subheader("Restored region (pure physics)" if pure_physics_mode else "Restored region")
    if run:
        spinner_text = (
            "Running closed-form Wiener deconvolution only (no learned denoising)..."
            if pure_physics_mode
            else "Running physics-informed deconvolution..."
        )
        with st.spinner(spinner_text):
            y = torch.from_numpy(crop_working).permute(2, 0, 1).unsqueeze(0)

            focal_length_t = torch.tensor([focal_length_mm / 1000.0])
            f_number_t = torch.tensor([f_number])
            pixel_pitch_t = torch.tensor([effective_pixel_pitch_um * 1e-6])
            focus_distance_t = torch.tensor([focus_distance_m])
            subject_distance_t = torch.tensor([subject_distance_m])

            # Quick geometric-only radius estimate, purely to size/cap the kernel
            # window for CPU-feasible processing before invoking the full
            # diffraction+dispersion engine (which derives its own, per-channel
            # radius from the raw camera parameters below).
            b_m = circle_of_confusion_diameter_m(
                focal_length_t, f_number_t, focus_distance_t, subject_distance_t
            )
            radius_px_raw = defocus_diameter_to_pixel_radius(b_m, pixel_pitch_t)
            radius_px = radius_px_raw.clamp(0.5, 25.0)
            ksize = int(2 * np.ceil(radius_px.item() * 1.6) + 1)
            was_clamped = abs(radius_px.item() - radius_px_raw.item()) > 1e-6

            kernel = model.build_kernel(
                focal_length_t, f_number_t, pixel_pitch_t, focus_distance_t, subject_distance_t, ksize=ksize
            )

            sigma_map = torch.full((1, 1, y.shape[-2], y.shape[-1]), float(sigma_val))

            with torch.no_grad():
                restored = model(
                    y, kernel, sigma_map, use_learned_prior=not pure_physics_mode
                ).clamp(0.0, 1.0)

        restored_np = restored.squeeze(0).permute(1, 2, 0).numpy()
        st.image(restored_np, use_container_width=True, clamp=True)
        st.caption(
            f"Blur kernel radius: {radius_px.item():.2f}px (kernel {ksize}x{ksize}, "
            f"per-RGB-channel diffraction + chromatic dispersion) | "
            f"CoC diameter: {b_m.item()*1e3:.3f} mm"
        )
        if was_clamped:
            st.warning(
                f"The physics-derived radius ({radius_px_raw.item():.1f}px) exceeded the "
                f"{radius_px.item():.0f}px cap and was clamped for computational feasibility. "
                "The parameters imply a stronger defocus than can be represented at this "
                "processing resolution -- try a smaller focus/subject distance gap, a "
                "larger pixel pitch value, or a smaller selection (less downscaling)."
            )

        restored_pil_native = Image.fromarray((restored_np * 255).astype(np.uint8))
        if crop_scale < 1.0:
            # Upsample back to the ORIGINAL selection size for compositing into the full
            # photo. This is purely for visual placement -- it does not add detail beyond
            # what was recovered at the (downscaled) processing resolution.
            restored_pil_for_composite = restored_pil_native.resize(
                (box["width"], box["height"]), Image.LANCZOS
            )
        else:
            restored_pil_for_composite = restored_pil_native

        mode_tag = "purephysics" if pure_physics_mode else "learned"

        buf_crop = io.BytesIO()
        restored_pil_native.save(buf_crop, format="PNG")
        st.download_button(
            "Download restored region only (PNG)",
            data=buf_crop.getvalue(),
            file_name=f"restored_region_{mode_tag}_{Path(uploaded.name).stem}.png",
            mime="image/png",
        )

        composite = pil_img.copy()
        composite.paste(restored_pil_for_composite, (box["left"], box["top"]))
        st.subheader(
            "Full photo with region restored (pure physics)"
            if pure_physics_mode
            else "Full photo with region restored"
        )
        st.image(np.asarray(composite), use_container_width=True)
        buf_full = io.BytesIO()
        composite.save(buf_full, format="PNG")
        st.download_button(
            "Download full photo with region restored (PNG)",
            data=buf_full.getvalue(),
            file_name=f"restored_full_{mode_tag}_{Path(uploaded.name).stem}.png",
            mime="image/png",
        )
    else:
        st.caption("Set parameters in the sidebar, then click **Run deblurring**.")
