# Camera Sensor Reference

Human-readable companion to [`camera_sensor_db.py`](camera_sensor_db.py), which is
the version this project's code actually reads. This document is for your own
reference — nothing here is consumed by the model. If you edit sensor data, edit
`camera_sensor_db.py`; update this file to match afterward.

## What this project actually needs from a camera, and why

The deblurring physics (`circle_of_confusion_diameter_m` in
[unfolded_optics_deblur.py](unfolded_optics_deblur.py)) needs exactly four numbers
per photo: focal length, f-number, sensor pixel pitch, and the focus/subject
distances. The first two come from every photo's own EXIF directly. Pixel pitch
comes from EXIF too when `FocalPlaneXResolution` is present (every Fujifilm file
tested so far has it) — the sensor-format table below exists **only** to cover
photos where that field is missing (confirmed missing on Nikon D60/D5100/D5300,
despite otherwise-complete EXIF). It is not a source of "extra training signal" by
itself — it fills a metadata gap so the physics can be computed correctly for
cameras whose EXIF is incomplete.

## Two things this table deliberately does NOT encode

**Mirror vs. mirrorless is not a physical variable here.** It describes the
viewfinder mechanism (an optical path through a reflex mirror vs. an electronic
sensor feed), not the sensor or the optics. A DSLR and a mirrorless body with the
same sensor format and the same lens produce identical defocus blur for a given
subject — the mirror plays no role in the circle-of-confusion formula. The table
below lists camera type for orientation only.

**"Optical resolution"** (megapixel count) isn't listed as a separate column,
because it's already read directly from each photo's own pixel dimensions — a
static table entry would just duplicate what every JPEG already tells us.

## Sensor formats (the physically meaningful grouping)

| Format | Width | Camera type | Notes |
|---|---|---|---|
| Medium format (44×33mm) | 43.8mm | Mirrorless (Fujifilm GFX, Hasselblad X) | Largest common format; shallowest DOF at a given f-number |
| Full-frame (35mm) | 36.0mm | DSLR or mirrorless | Reference format most f-number/DOF intuition is built around |
| APS-H | 27.9mm | DSLR (legacy Canon 1D-series) | Rare today |
| APS-C (Canon) | 22.3mm | DSLR or mirrorless | ~1.6x crop vs. full-frame |
| APS-C (Nikon/Sony/Fuji/Pentax, "DX") | 23.5mm | DSLR or mirrorless | ~1.5x crop; Fujifilm X-Trans uses this size but usually has real focal-plane EXIF |
| Micro Four Thirds | 17.3mm | Mirrorless only | No MFT DSLRs exist; 2.0x crop |
| 1-inch | 13.2mm | Fixed-lens compact / drone | Sony RX100 series, most consumer drones |
| 1/1.7" | 7.44mm | Compact | Older premium compacts |
| 1/2.3" | 6.16mm | Compact / action cam / phone | Budget compacts, action cams, older phones |
| Phone ~1/1.28" | 9.8mm | Smartphone (approximate) | Large recent flagship main sensor, often pixel-binned |
| Phone ~1/1.3" | 9.6mm | Smartphone (approximate) | |
| Phone ~1/1.56" | 8.0mm | Smartphone (approximate) | Mid-size flagship main sensor |
| Phone ~1/2.55" | 5.7mm | Smartphone (approximate) | Smaller main/ultrawide module |

## Known problems and strengths, by format (general photography knowledge, not project-specific)

- **Medium format**: exceptional detail and tonal range; expensive, slower to shoot, shallow DOF can make focus-critical work harder.
- **Full-frame**: strong low-light/high-ISO performance and shallow-DOF control; larger, heavier, pricier bodies and lenses.
- **APS-C**: a common sweet spot of cost, size, and image quality; smaller sensor means deeper DOF at a given f-number/framing than full-frame (harder to get creamy background blur), and typically weaker high-ISO noise performance than full-frame.
- **Micro Four Thirds**: smallest interchangeable-lens format in common use, enabling very compact bodies/lenses and excellent in-body stabilization headroom; the smaller sensor area gives the deepest DOF and weakest low-light performance of the interchangeable-lens formats.
- **1-inch and smaller compacts**: convenience and pocketability; progressively worse low-light noise and dynamic range as the sensor shrinks.
- **Smartphone sensors**: tiny individual photosites, compensated by aggressive computational photography (multi-frame HDR fusion, "Night Mode" stacking, AI-driven sharpening/denoising) applied *before* the JPEG is saved. This is the single biggest caveat for this project: our deblurring model assumes one optical exposure blurred by one PSF plus sensor noise, which is a much weaker description of a phone's already-fused, already-processed output than of a single DSLR/mirrorless exposure. A phone photo's pixel pitch being in this table doesn't mean the overall physical model fits it as well.

## DSLR vs. mirrorless, for completeness

| | Mechanism | Relevant to our physics? |
|---|---|---|
| DSLR | Reflex mirror bounces light to an optical viewfinder; flips up during exposure | No — sensor and lens are what matter, not the mirror |
| Mirrorless | Sensor feeds an electronic viewfinder/screen directly; no mirror | No — same reasoning |

Both types exist across every interchangeable-lens sensor format above. Knowing a
camera is "a DSLR" or "mirrorless" tells you nothing about its pixel pitch — you
need the sensor format specifically, which is what `camera_sensor_db.py` maps to.

## Extending this

To add a camera: find its sensor format (manufacturer spec sheets almost always
state it, e.g. "APS-C" or "Four Thirds"), then add one line to
`CAMERA_MODEL_TO_FORMAT` in `camera_sensor_db.py` mapping its exact
`(Make, Model)` EXIF strings to that format's key. You only need to do this for
cameras that omit `FocalPlaneXResolution` — check with:

```bash
python exif_utils.py path/to/photo.jpg
```

If `pixel_pitch_source` comes back `exif_focal_plane`, the camera already reports
it directly and needs no table entry.
