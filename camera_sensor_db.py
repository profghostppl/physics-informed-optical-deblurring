"""
Camera sensor width database, used by exif_utils.py as a fallback ONLY when a photo's
own EXIF lacks FocalPlaneXResolution (the per-photo reading exif_utils.py prefers
whenever it's present -- confirmed present on every Fujifilm file tested so far,
confirmed ABSENT on Nikon D60/D5100/D5300 despite those files having otherwise full
EXIF). This module exists to extend that fallback beyond the 3 models hardcoded
earlier, toward any camera -- DSLR, mirrorless, or smartphone -- that might show up.

Design: two tiers, because sensor width is fundamentally a property of the SENSOR
FORMAT (a small, standardized set shared across many camera models), not of each
individual model. Cataloging every camera ever made directly would be enormous and
perpetually incomplete; mapping models to their (shared) format is far more tractable
and is how the photography industry itself organizes this information.

  1. SENSOR_FORMATS: format name -> width in meters. These are standardized,
     well-published physical dimensions (the DSLR/mirrorless entries are exact
     industry-standard specs; the smartphone entries are the least standardized and
     are explicitly flagged as approximate).
  2. CAMERA_MODEL_TO_FORMAT: (Make, Model) -> format name, exactly as those two EXIF
     tags read after exif_utils.py's null-byte/whitespace cleaning. Extend this
     incrementally as new cameras show up; it only needs entries for cameras that
     (like the Nikons above) omit focal-plane EXIF.

Two honest limitations, spelled out here rather than left implicit:

  * Mirror vs. mirrorless is NOT a variable in this database, deliberately. It
    describes the viewfinder mechanism, not the sensor. A DSLR and a mirrorless
    camera with the same sensor format and the same lens produce IDENTICAL defocus
    physics -- the circle-of-confusion formula this project uses depends only on
    focal length, f-number, sensor pixel pitch, and distances, none of which the
    mirror affects. SENSOR_FORMAT_NOTES below records camera *type* for reference
    (DSLR/mirrorless/compact/phone), but the physics code never reads it.

  * Smartphone sensor sizes are NOT read from a per-photo EXIF field the way
    Fujifilm's focal-plane resolution is -- they vary continuously across models,
    frequently use pixel-binning (a "50MP" sensor may deliver 12.5MP binned output,
    at a correspondingly different effective pixel pitch than the raw photosite
    count implies), and, more fundamentally, nearly every phone photo is the
    product of multi-frame computational fusion (HDR stacking, Night Mode, "Deep
    Fusion"-style pipelines) BEFORE the JPEG is saved. This project's core physical
    model (y = Kx + n, a single optical exposure blurred by one PSF plus sensor
    noise) is a much weaker fit for a computationally-fused phone photo than for a
    single DSLR/mirrorless exposure. Phone entries below are included because an
    approximately-right pixel pitch is still better than none, not because the
    overall physical model is expected to hold as well.
"""

from __future__ import annotations

from typing import Optional

# ----------------------------------------------------------------------------------
# Tier 1: sensor formats -> width in meters.
# DSLR/mirrorless values are exact, widely published industry-standard specs.
# Smartphone/compact values are approximate (see module docstring) -- flagged below.
# ----------------------------------------------------------------------------------
SENSOR_FORMATS_M: dict[str, float] = {
    # -- Interchangeable-lens formats (DSLR + mirrorless share these) --
    "medium_format_44x33": 43.8e-3,       # Fujifilm GFX, Hasselblad X
    "full_frame_35mm": 36.0e-3,           # Nikon FX, Canon FF, Sony FE, Panasonic S
    "aps_h": 27.9e-3,                     # Canon 1D-series (legacy)
    "aps_c_canon": 22.3e-3,               # Canon APS-C (1.6x crop)
    "aps_c_nikon_sony_fuji_pentax": 23.5e-3,  # "DX"/APS-C, ~1.5x crop
    "micro_four_thirds": 17.3e-3,         # Olympus/OM System, Panasonic G-series

    # -- Fixed-lens / compact formats --
    "one_inch": 13.2e-3,                  # Sony RX100 series, many drones
    "one_over_1_7_inch": 7.44e-3,         # premium compacts (older Canon G-series etc.)
    "one_over_2_3_inch": 6.16e-3,         # superzooms, action cams, older phones

    # -- Smartphone main-camera formats (APPROXIMATE -- see docstring caveats) --
    "phone_one_over_1_28_inch": 9.8e-3,   # large flagship phone main sensors (~2020s)
    "phone_one_over_1_3_inch": 9.6e-3,
    "phone_one_over_1_56_inch": 8.0e-3,
    "phone_one_over_2_55_inch": 5.7e-3,   # smaller phone main/ultrawide sensors
}

# Type notes are for human reference only -- never consulted by the physics code.
SENSOR_FORMAT_NOTES: dict[str, str] = {
    "medium_format_44x33": "Medium format mirrorless. Largest common format; shallowest DOF at a given f-number/frame.",
    "full_frame_35mm": "DSLR or mirrorless. The reference format most f-number/DOF intuition is built around.",
    "aps_h": "DSLR (legacy Canon 1D-series). Rare today.",
    "aps_c_canon": "DSLR or mirrorless. ~1.6x crop vs full-frame; slightly smaller than other brands' APS-C.",
    "aps_c_nikon_sony_fuji_pentax": "DSLR or mirrorless. ~1.5x crop vs full-frame. Fujifilm's X-Trans sensors use this size but usually DO report focal-plane EXIF directly.",
    "micro_four_thirds": "Mirrorless only (no MFT DSLRs were made). 2.0x crop vs full-frame.",
    "one_inch": "Fixed-lens compacts and most consumer drones.",
    "one_over_1_7_inch": "Older premium compacts.",
    "one_over_2_3_inch": "Budget compacts, action cameras, older/budget phones.",
    "phone_one_over_1_28_inch": "Large recent flagship phone main sensor. Often pixel-binned (e.g. quad-Bayer) -- delivered JPEG resolution may not equal raw photosite count.",
    "phone_one_over_1_3_inch": "Large phone main sensor, pixel-binned in many models.",
    "phone_one_over_1_56_inch": "Mid-size flagship phone main sensor.",
    "phone_one_over_2_55_inch": "Smaller phone sensor, typical of ultrawide/telephoto modules.",
}

# ----------------------------------------------------------------------------------
# Tier 2: (Make, Model) -> format name, exactly as EXIF reports them after
# exif_utils.py's _clean_str() (null bytes stripped, whitespace trimmed).
# Only needs entries for cameras that omit focal-plane EXIF; harmless to list others.
# ----------------------------------------------------------------------------------
CAMERA_MODEL_TO_FORMAT: dict[tuple[str, str], str] = {
    # --- Nikon DSLRs (mirror) ---
    ("NIKON CORPORATION", "NIKON D40"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D60"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D90"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D300S"): "aps_c_nikon_sony_fuji_pentax",  # 2009 semi-pro DSLR, 23.6x15.8mm DX
    ("NIKON CORPORATION", "NIKON D3000"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D5000"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D5100"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D5200"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D5300"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D5500"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D5600"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D7000"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D7100"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D7200"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D7500"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON D600"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D610"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D700"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D750"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D780"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D800"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D810"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D850"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D3"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D4"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D5"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON D6"): "full_frame_35mm",
    # --- Nikon Z mirrorless ---
    ("NIKON CORPORATION", "NIKON Z30"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON Z50"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON Z fc"): "aps_c_nikon_sony_fuji_pentax",
    ("NIKON CORPORATION", "NIKON Z5"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON Z6"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON Z6_2"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON Z6III"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON Z7"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON Z7_2"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON Z8"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON Z9"): "full_frame_35mm",
    ("NIKON CORPORATION", "NIKON Zf"): "full_frame_35mm",

    # --- Canon DSLRs (mirror) ---
    ("Canon", "Canon EOS REBEL T3i"): "aps_c_canon",
    ("Canon", "Canon EOS REBEL T5i"): "aps_c_canon",
    ("Canon", "Canon EOS REBEL T6i"): "aps_c_canon",
    ("Canon", "Canon EOS REBEL T7i"): "aps_c_canon",
    ("Canon", "Canon EOS REBEL T8i"): "aps_c_canon",
    ("Canon", "Canon EOS 60D"): "aps_c_canon",
    ("Canon", "Canon EOS 70D"): "aps_c_canon",
    ("Canon", "Canon EOS 80D"): "aps_c_canon",
    ("Canon", "Canon EOS 90D"): "aps_c_canon",
    ("Canon", "Canon EOS 7D"): "aps_c_canon",
    ("Canon", "Canon EOS 7D Mark II"): "aps_c_canon",
    ("Canon", "Canon EOS 5D"): "full_frame_35mm",
    ("Canon", "Canon EOS 5D Mark II"): "full_frame_35mm",
    ("Canon", "Canon EOS 5D Mark III"): "full_frame_35mm",
    ("Canon", "Canon EOS 5D Mark IV"): "full_frame_35mm",
    ("Canon", "Canon EOS 6D"): "full_frame_35mm",
    ("Canon", "Canon EOS 6D Mark II"): "full_frame_35mm",
    ("Canon", "Canon EOS-1D X"): "full_frame_35mm",
    ("Canon", "Canon EOS-1D X Mark II"): "full_frame_35mm",
    # --- Canon EOS R mirrorless ---
    ("Canon", "Canon EOS M50"): "aps_c_canon",
    ("Canon", "Canon EOS R7"): "aps_c_canon",
    ("Canon", "Canon EOS R10"): "aps_c_canon",
    ("Canon", "Canon EOS R50"): "aps_c_canon",
    ("Canon", "Canon EOS R100"): "aps_c_canon",
    ("Canon", "Canon EOS R"): "full_frame_35mm",
    ("Canon", "Canon EOS RP"): "full_frame_35mm",
    ("Canon", "Canon EOS R5"): "full_frame_35mm",
    ("Canon", "Canon EOS R6"): "full_frame_35mm",
    ("Canon", "Canon EOS R6m2"): "full_frame_35mm",
    ("Canon", "Canon EOS R3"): "full_frame_35mm",
    ("Canon", "Canon EOS R8"): "full_frame_35mm",

    # --- Sony mirrorless ---
    ("SONY", "ILCE-6000"): "aps_c_nikon_sony_fuji_pentax",
    ("SONY", "ILCE-6100"): "aps_c_nikon_sony_fuji_pentax",
    ("SONY", "ILCE-6300"): "aps_c_nikon_sony_fuji_pentax",
    ("SONY", "ILCE-6400"): "aps_c_nikon_sony_fuji_pentax",
    ("SONY", "ILCE-6500"): "aps_c_nikon_sony_fuji_pentax",
    ("SONY", "ILCE-6600"): "aps_c_nikon_sony_fuji_pentax",
    ("SONY", "ILCE-6700"): "aps_c_nikon_sony_fuji_pentax",
    ("SONY", "ILCE-7"): "full_frame_35mm",
    ("SONY", "ILCE-7M2"): "full_frame_35mm",
    ("SONY", "ILCE-7M3"): "full_frame_35mm",
    ("SONY", "ILCE-7M4"): "full_frame_35mm",
    ("SONY", "ILCE-7RM3"): "full_frame_35mm",
    ("SONY", "ILCE-7RM4"): "full_frame_35mm",
    ("SONY", "ILCE-7RM5"): "full_frame_35mm",
    ("SONY", "ILCE-7SM3"): "full_frame_35mm",
    ("SONY", "ILCE-9"): "full_frame_35mm",
    ("SONY", "ILCE-1"): "full_frame_35mm",

    # --- Fujifilm (usually has real focal-plane EXIF; listed as a safety-net fallback) ---
    ("FUJIFILM", "X-T30"): "aps_c_nikon_sony_fuji_pentax",
    ("FUJIFILM", "X-T3"): "aps_c_nikon_sony_fuji_pentax",
    ("FUJIFILM", "X-T4"): "aps_c_nikon_sony_fuji_pentax",
    ("FUJIFILM", "X-T5"): "aps_c_nikon_sony_fuji_pentax",
    ("FUJIFILM", "X-Pro2"): "aps_c_nikon_sony_fuji_pentax",
    ("FUJIFILM", "X-Pro3"): "aps_c_nikon_sony_fuji_pentax",
    ("FUJIFILM", "X100V"): "aps_c_nikon_sony_fuji_pentax",
    ("FUJIFILM", "X-S10"): "aps_c_nikon_sony_fuji_pentax",
    ("FUJIFILM", "X-S20"): "aps_c_nikon_sony_fuji_pentax",
    ("FUJIFILM", "GFX 50S"): "medium_format_44x33",
    ("FUJIFILM", "GFX 100"): "medium_format_44x33",
    ("FUJIFILM", "GFX100S"): "medium_format_44x33",

    # --- Panasonic / Olympus / OM System (Micro Four Thirds, mirrorless only) ---
    ("OLYMPUS CORPORATION", "E-M1MarkII"): "micro_four_thirds",
    ("OLYMPUS CORPORATION", "E-M5MarkIII"): "micro_four_thirds",
    ("OLYMPUS CORPORATION", "E-M10MarkIV"): "micro_four_thirds",
    ("OM Digital Solutions", "OM-1"): "micro_four_thirds",
    ("OM Digital Solutions", "OM-5"): "micro_four_thirds",
    ("Panasonic", "DC-GH5"): "micro_four_thirds",
    ("Panasonic", "DC-GH6"): "micro_four_thirds",
    ("Panasonic", "DC-G9"): "micro_four_thirds",
    ("Panasonic", "DC-GX85"): "micro_four_thirds",

    # --- Pentax DSLRs (mirror) ---
    ("PENTAX", "PENTAX K-70"): "aps_c_nikon_sony_fuji_pentax",
    ("PENTAX", "PENTAX K-3"): "aps_c_nikon_sony_fuji_pentax",
    ("PENTAX", "PENTAX KP"): "aps_c_nikon_sony_fuji_pentax",
    ("PENTAX", "PENTAX K-1"): "full_frame_35mm",
    ("PENTAX", "PENTAX K-1 Mark II"): "full_frame_35mm",

    # --- Smartphones (APPROXIMATE main-camera sensor -- see module docstring) ---
    ("Apple", "iPhone 12 Pro"): "phone_one_over_2_55_inch",
    ("Apple", "iPhone 13 Pro"): "phone_one_over_1_56_inch",
    ("Apple", "iPhone 14 Pro"): "phone_one_over_1_28_inch",
    ("Apple", "iPhone 15 Pro"): "phone_one_over_1_28_inch",
    ("Apple", "iPhone 16 Pro"): "phone_one_over_1_28_inch",
    ("samsung", "SM-G991B"): "phone_one_over_2_55_inch",   # Galaxy S21
    ("samsung", "SM-S911B"): "phone_one_over_1_56_inch",   # Galaxy S23
    ("samsung", "SM-S928B"): "phone_one_over_1_3_inch",    # Galaxy S24 Ultra (200MP sensor, pixel-binned)
    ("Google", "Pixel 7 Pro"): "phone_one_over_1_3_inch",
    ("Google", "Pixel 8 Pro"): "phone_one_over_1_3_inch",
}


def lookup_sensor_width_m(make: Optional[str], model: Optional[str]) -> Optional[float]:
    """(Make, Model) -> published sensor width in meters, or None if not catalogued."""
    if not make or not model:
        return None
    fmt = CAMERA_MODEL_TO_FORMAT.get((make, model))
    if fmt is None:
        return None
    return SENSOR_FORMATS_M.get(fmt)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python camera_sensor_db.py <Make> <Model>")
        sys.exit(1)
    w = lookup_sensor_width_m(sys.argv[1], sys.argv[2])
    print(f"Sensor width: {w*1000:.2f} mm" if w else "Not found in database.")
