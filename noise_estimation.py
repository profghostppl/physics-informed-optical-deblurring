"""
Fast, classical (non-learned) noise-level estimator, used to auto-populate the
sigma input the network expects, rather than defaulting it to an arbitrary constant.

Implements J. Immerkaer, "Fast Noise Variance Estimation", Computer Vision and Image
Understanding, 1996: convolve with a discrete Laplacian-of-a-bilinear-surface mask
that is insensitive to smooth image structure but responsive to i.i.d. noise, and
rescale the mean absolute response into a noise standard deviation.
"""

import numpy as np
from scipy.signal import convolve2d

_LAPLACIAN_MASK = np.array(
    [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64
)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """
    Args:
        image: HxW or HxWxC array, float in [0,1] (or uint8 in [0,255]; auto-detected).
    Returns:
        Estimated noise standard deviation, in the same [0,1] scale used by the model.
    """
    img = image.astype(np.float64)
    if img.max() > 1.5:  # heuristically detect uint8-range input
        img = img / 255.0
    if img.ndim == 3:
        img = img.mean(axis=-1)

    h, w = img.shape
    if h < 5 or w < 5:
        return 0.01  # degenerate/too-small input; fall back to a mild default

    response = convolve2d(img, _LAPLACIAN_MASK, mode="valid")
    sigma = np.sqrt(np.pi / 2.0) * np.sum(np.abs(response)) / (6.0 * (w - 2) * (h - 2))
    return float(np.clip(sigma, 1e-4, 0.3))


if __name__ == "__main__":
    import sys
    from skimage import io as skio

    if len(sys.argv) != 2:
        print("Usage: python noise_estimation.py <image_path>")
        sys.exit(1)
    im = skio.imread(sys.argv[1])
    print(f"Estimated noise sigma: {estimate_noise_sigma(im):.5f}")
