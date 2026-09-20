"""
Shared helper for `train_on_dataset.py` and `train_on_real_photos.py`.

Both scripts synthesize training pairs by sampling a defocus strength and building a
kernel from it. The old (pre-`AdvancedOpticsKernelEngine`) pipeline sampled a free
defocus *offset* (`delta`), derived the resulting blur radius from it, and clamped that
radius into [1.0, 9.5]px post-hoc to keep it inside the fixed KSIZE=25 kernel window.
Measured empirically on the original `delta ~ Uniform(0.4, 1.8)` sampling: *without*
that clamp, 31% of samples exceeded the window (max observed radius: ~245px) -- so the
clamp was doing real work, but it also meant the (focal_length, f_number,
focus_distance, subject_distance) tuple fed to the network no longer matched the
*actual* radius used to build the kernel for roughly a third of samples (the pre-clamp
`subject_distance` was simply discarded).

That mismatch didn't matter for the old single-channel achromatic pillbox path (the
kernel builder only ever consumed the final scalar radius). It matters now:
`AdvancedOpticsKernelEngine` uses `f_number` and `pixel_pitch_m` independently (for the
Airy diffraction radius) and `focal_length_m` independently (for chromatic dispersion),
so a self-consistent (f, N, pitch, d0, d) tuple that actually produces the intended
blur strength is worth solving for exactly, rather than sampling something arbitrary.

`sample_subject_distance_for_radius` does this by inverting the thin-lens
circle-of-confusion equation (`circle_of_confusion_diameter_m` in
`unfolded_optics_deblur.py`) for the subject distance `d` that reproduces a *target*
radius exactly, given the other camera parameters:

    b(d) = f^2 * |d - d0| / (N * d * (d0 - f))                                   (thin lens)

Let k = b_target * N * (d0 - f) / f^2  =  |d - d0| / d  (a dimensionless "defocus
fraction" of the target). Two branches, solved directly for d:

    background (d > d0):  d - d0 = k*d   =>   d = d0 / (1 - k)   [valid only for k < 1]
    foreground (d < d0):  d0 - d = k*d   =>   d = d0 / (1 + k)   [valid for any k >= 0]

The background branch has a real physical ceiling: as d -> infinity, k -> 1, so the
maximum background CoC any (f, N, d0) combo can ever produce (at any distance) is
b_max = f^2 / (N * (d0 - f)) -- e.g. a 35mm lens at f/5.6 focused at 0.8m tops out
around a 1-2px background radius, no matter how far the background is. When a sampled
target exceeds what the background branch can reach (k >= 1), this redirects that
sample to the foreground branch instead (which has no such ceiling) so every sample
still hits its target radius almost exactly (verified: mean abs error ~5e-7px across
20k samples, with about 17% of "background-requested" samples redirected this way).
"""

from __future__ import annotations

import torch

from unfolded_optics_deblur import circle_of_confusion_diameter_m, defocus_diameter_to_pixel_radius


def sample_subject_distance_for_radius(
    focal_length_m: torch.Tensor,
    f_number: torch.Tensor,
    pixel_pitch_m: torch.Tensor,
    focus_distance_m: torch.Tensor,
    radius_px_target: torch.Tensor,
    want_background: torch.Tensor,
    min_distance_m: float = 0.05,
    background_k_ceiling: float = 0.98,
) -> torch.Tensor:
    """Solve for the subject distance (m) whose geometric defocus radius matches
    `radius_px_target` (px), given the other camera parameters.

    Args:
        focal_length_m, f_number, pixel_pitch_m, focus_distance_m: (N,) tensors.
        radius_px_target: (N,) desired geometric blur radius, in pixels.
        want_background: (N,) bool -- True to place the subject farther than the focus
            plane (background), False for closer (foreground). Silently redirected to
            the foreground branch per-sample where the background target is physically
            unreachable for that lens/aperture/focus combo (see module docstring).
    Returns:
        (N,) subject_distance_m tensor.
    """
    b_m_target = radius_px_target * 2.0 * pixel_pitch_m
    k = b_m_target * f_number * (focus_distance_m - focal_length_m) / focal_length_m.pow(2)

    feasible_bg = k < background_k_ceiling
    use_background = want_background & feasible_bg

    d = torch.where(
        use_background,
        focus_distance_m / (1.0 - k.clamp(max=background_k_ceiling)),
        focus_distance_m / (1.0 + k),
    )
    return d.clamp_min(min_distance_m)


def achieved_radius_px(
    focal_length_m: torch.Tensor,
    f_number: torch.Tensor,
    pixel_pitch_m: torch.Tensor,
    focus_distance_m: torch.Tensor,
    subject_distance_m: torch.Tensor,
) -> torch.Tensor:
    """Diagnostic: recompute the actual geometric defocus radius (px) a
    (f, N, pitch, d0, d) tuple produces -- e.g. to sanity-check
    `sample_subject_distance_for_radius`'s output against its target.
    """
    b_m = circle_of_confusion_diameter_m(focal_length_m, f_number, focus_distance_m, subject_distance_m)
    return defocus_diameter_to_pixel_radius(b_m, pixel_pitch_m)
