"""
Camera metadata extraction from standard EXIF tags.

What this can and cannot honestly give you
--------------------------------------------
* Focal length (f) and f-number/aperture (N): standard EXIF fields
  (`FocalLength`, `FNumber`), present on almost all camera-shot JPEGs. Reliable.

* Sensor pixel pitch (p): NOT usually stored directly, but derivable three ways,
  tried in order of decreasing precision:
    1. `FocalPlaneXResolution` / `FocalPlaneYResolution` + `FocalPlaneResolutionUnit`
       + the image's pixel dimensions -- these three together give the sensor's
       physical width directly, no camera-model lookup table needed. Present on many
       cameras (all Fujifilm bodies tested so far) but genuinely ABSENT from others
       -- e.g. older Nikon DSLRs (D60, D5100) never write these tags at all,
       confirmed by inspecting their raw EXIF.
    2. A small table of publicly published sensor widths for specific (Make, Model)
       pairs (`camera_sensor_db.py`) -- an exact documented hardware spec, but only
       covers models someone has explicitly added to the table.
    3. The standard `FocalLengthIn35mmFilm` tag (35mm-equivalent focal length),
       divided by the actual `FocalLength` to get the lens's crop factor, then
       `36mm / crop_factor` for sensor width. This tag is present on nearly every
       camera with any auto/program exposure mode (required for flash-metering
       compatibility) -- including cameras that omit `FocalPlaneXResolution`
       entirely -- and needs no per-model database entry. Less precise than tier 2
       (both source fields are typically camera-rounded to the nearest mm), so it's
       only tried after an exact per-model lookup fails.
  Each tier is reported with a distinct `pixel_pitch_source` so callers can show its
  confidence. Cameras that fail all three get `pixel_pitch_m = None`.

* Focus distance (d0) and subject distance (d): there IS a standard `SubjectDistance`
  EXIF tag, but in practice most manufacturers leave it unpopulated, hardcode it to
  0, or only expose an approximate value inside proprietary MakerNote binary blobs
  (e.g. Canon/Nikon-specific formats) that require vendor-specific parsers such as
  `exiftool`, not a generic EXIF reader. This module reports `SubjectDistance` when
  present but flags it as low-confidence, and otherwise returns None -- callers
  should treat d0/d as user-supplied, not silently guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, BinaryIO

from PIL import Image, ExifTags

from camera_sensor_db import lookup_sensor_width_m, SENSOR_FORMATS_M

_EXIF_IFD_TAG = 0x8769  # "Exif IFD Pointer" -- FocalLength/FNumber/FocalPlane* live here

# EXIF ResolutionUnit / FocalPlaneResolutionUnit codes -> meters per unit.
_UNIT_TO_METERS = {
    2: 0.0254,   # inch
    3: 0.01,     # centimeter
    4: 0.001,    # millimeter (nonstandard but seen in the wild)
    5: 1e-6,     # micrometer (nonstandard but seen in the wild)
}
# Fallback sensor-width lookup (by camera format, not per-photo EXIF) for cameras
# that never write FocalPlaneXResolution -- see camera_sensor_db.py for the full
# table and its caveats (mirror-vs-mirrorless irrelevance, smartphone approximations).
# Note: pitch is computed as sensor_width / THIS FILE's actual pixel width, not the
# camera's native resolution -- if a photo was saved at reduced JPEG size in-camera,
# each of its pixels genuinely spans more of the physical sensor, so the resulting
# pitch is correctly larger, not an error.


@dataclass
class CameraMetadata:
    make: Optional[str] = None
    model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    f_number: Optional[float] = None
    pixel_pitch_m: Optional[float] = None
    pixel_pitch_source: Optional[str] = None       # "exif_focal_plane" |
                                                     # "known_sensor_fallback" |
                                                     # "exif_35mm_crop_factor" | None
    subject_distance_m: Optional[float] = None      # low-confidence when present
    subject_distance_confidence: str = "none"        # "none" | "low"
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    has_exif: bool = False


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def extract_camera_metadata(image_source: Union[str, BinaryIO]) -> CameraMetadata:
    """Read what EXIF can honestly tell us about the capturing camera/lens.

    Args:
        image_source: file path or a file-like object (e.g. an uploaded file buffer).
    Returns:
        CameraMetadata with whichever fields EXIF actually provided; everything else
        is left as None for the caller (e.g. a UI) to ask the user for explicitly.
    """
    meta = CameraMetadata()
    try:
        img = Image.open(image_source)
        meta.image_width, meta.image_height = img.size
        exif = img.getexif()
        if not exif:
            return meta

        base = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        try:
            sub_ifd = exif.get_ifd(_EXIF_IFD_TAG)
            sub = {ExifTags.TAGS.get(k, k): v for k, v in sub_ifd.items()}
        except Exception:
            sub = {}
        tags = {**base, **sub}
        if not tags:
            return meta

        def _clean_str(v) -> Optional[str]:
            # Some vendors (e.g. Fujifilm) null-pad fixed-width string tags.
            s = str(v).replace("\x00", "").strip()
            return s or None

        meta.has_exif = True
        meta.make = _clean_str(tags.get("Make", ""))
        meta.model = _clean_str(tags.get("Model", ""))
        meta.lens_model = _clean_str(tags.get("LensModel", ""))

        meta.focal_length_mm = _to_float(tags.get("FocalLength"))
        meta.f_number = _to_float(tags.get("FNumber"))

        # Physical sensor width, and therefore pixel pitch, from focal-plane resolution.
        fp_x_res = _to_float(tags.get("FocalPlaneXResolution"))
        fp_unit = tags.get("FocalPlaneResolutionUnit")
        exif_width = tags.get("PixelXDimension") or meta.image_width
        if fp_x_res and fp_x_res > 0 and exif_width:
            meters_per_unit = _UNIT_TO_METERS.get(int(fp_unit) if fp_unit else 2, 0.0254)
            meta.pixel_pitch_m = meters_per_unit / fp_x_res
            meta.pixel_pitch_source = "exif_focal_plane"
        elif meta.image_width:
            sensor_width_m = lookup_sensor_width_m(meta.make, meta.model)
            if sensor_width_m:
                meta.pixel_pitch_m = sensor_width_m / meta.image_width
                meta.pixel_pitch_source = "known_sensor_fallback"
            else:
                # Third fallback: derive sensor width from the standard 35mm-
                # equivalent crop factor. No per-camera-model database entry
                # needed -- see the module docstring for why this is tried last.
                focal_length_35mm = _to_float(tags.get("FocalLengthIn35mmFilm"))
                if focal_length_35mm and focal_length_35mm > 0 and meta.focal_length_mm:
                    crop_factor = focal_length_35mm / meta.focal_length_mm
                    sensor_width_m = SENSOR_FORMATS_M["full_frame_35mm"] / crop_factor
                    meta.pixel_pitch_m = sensor_width_m / meta.image_width
                    meta.pixel_pitch_source = "exif_35mm_crop_factor"

        # SubjectDistance: standard tag, but notoriously unreliable across vendors.
        subj_dist = _to_float(tags.get("SubjectDistance"))
        if subj_dist and subj_dist > 0:
            meta.subject_distance_m = subj_dist
            meta.subject_distance_confidence = "low"

    except Exception:
        # Corrupt/absent EXIF, unsupported format, etc. -- fail soft, return defaults.
        pass

    return meta


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python exif_utils.py <image_path>")
        sys.exit(1)
    m = extract_camera_metadata(sys.argv[1])
    print(m)
