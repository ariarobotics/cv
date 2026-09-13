"""Homework 02: Filtering.

Complete the three TODO functions below. Do not change their names or
arguments; the Gradescope autograder calls them directly.

Run locally from the course `cv` Conda environment with:
    python hw02.py my_image.png

The script will save a visualization to hw02_output.png. You ONLY submit
hw02.py to Gradescope.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


BOX_3X3 = np.ones((3, 3), dtype=np.float32) / 9.0

SOBEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)

SHARPEN_3X3 = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)


def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return same-size 2-D correlation with zero padding.

    Requirements:
    - `image` and `kernel` are 2-D arrays.
    - The kernel has odd height and width.
    - Use zero padding so the output has the same shape as the input.
    - Apply the kernel as written; do NOT rotate it.
    - Implement the neighborhood sweep yourself using nested `for` loops.
    - Do not call a library filtering/correlation/convolution routine.
    - Return a NumPy array with dtype np.float32.
    """
    # TODO
    raise NotImplementedError


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return same-size 2-D convolution with zero padding.

    Rotate the kernel by 180 degrees, then reuse correlate2d(). You may use
    np.flip() or cv2.flip() to rotate the kernel; the filtering itself must
    still be performed by your correlate2d() implementation.
    """
    # TODO
    raise NotImplementedError


def median_filter3x3(image: np.ndarray) -> np.ndarray:
    """Return a same-size 3x3 median-filtered image with zero padding.

    Sweep the image yourself using nested `for` loops. You may use np.median()
    to compute the median of each local 3x3 neighborhood, but you may NOT use
    cv2.medianBlur(), scipy.ndimage.median_filter(), or another library median
    filtering implementation. Return dtype np.float32.
    """
    # TODO
    raise NotImplementedError


# -----------------------------------------------------------------------------
# The helper functions below are provided. You do not need to modify them.
# -----------------------------------------------------------------------------

def load_grayscale_float(filename: str | Path) -> np.ndarray:
    """Load an image with OpenCV and return grayscale float32 in [0, 1]."""
    image = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {filename}")
    return image.astype(np.float32) / 255.0


def resize_for_filtering(image: np.ndarray, max_dimension: int = 96) -> np.ndarray:
    """Downsize large images so the manual Python loops run quickly."""
    height, width = image.shape
    largest = max(height, width)
    if largest <= max_dimension:
        return image
    scale = max_dimension / largest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def add_salt_and_pepper_noise(
    image: np.ndarray, probability: float = 0.08, seed: int = 7
) -> np.ndarray:
    """Create deterministic salt-and-pepper noise for the visualization."""
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    random_values = rng.random(image.shape)
    noisy[random_values < probability / 2] = 0.0
    noisy[random_values > 1.0 - probability / 2] = 1.0
    return noisy.astype(np.float32)


def _to_display_u8(image: np.ndarray, signed_response: bool = False) -> np.ndarray:
    """Convert a float image into a viewable uint8 image for the montage."""
    arr = np.asarray(image, dtype=np.float32)
    if signed_response:
        arr = np.abs(arr)

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        scaled = np.zeros_like(arr, dtype=np.float32)
    elif lo >= 0.0 and hi <= 1.0:
        scaled = arr
    else:
        scaled = (arr - lo) / (hi - lo)

    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def save_montage(
    panels: list[tuple[str, np.ndarray, bool]],
    output_path: str | Path = "hw02_output.png",
) -> None:
    """Save a 2x3 OpenCV montage of filtering results."""
    rendered = []
    for label, image, signed_response in panels:
        panel = _to_display_u8(image, signed_response)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 20), (255, 255, 255), -1)
        cv2.putText(
            panel,
            label,
            (4, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        rendered.append(panel)

    row1 = cv2.hconcat(rendered[:3])
    row2 = cv2.hconcat(rendered[3:6])
    montage = cv2.vconcat([row1, row2])
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"Could not save {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HW02: Filtering")
    parser.add_argument("image", help="Path to a JPG/PNG image (you may reuse your HW01 headshot)")
    args = parser.parse_args()

    image = resize_for_filtering(load_grayscale_float(args.image))

    box_corr = correlate2d(image, BOX_3X3)
    box_conv = convolve2d(image, BOX_3X3)
    sobel_corr = correlate2d(image, SOBEL_X)
    sobel_conv = convolve2d(image, SOBEL_X)
    sharpened = correlate2d(image, SHARPEN_3X3)

    noisy = add_salt_and_pepper_noise(image)
    median = median_filter3x3(noisy)

    print(
        "Box correlation vs convolution max difference:",
        float(np.max(np.abs(box_corr - box_conv))),
    )
    print(
        "Sobel-X correlation vs convolution max difference:",
        float(np.max(np.abs(sobel_corr - sobel_conv))),
    )

    save_montage(
        [
            ("Original", image, False),
            ("Box 3x3", box_corr, False),
            ("|Sobel X|", sobel_corr, True),
            ("Sharpen", sharpened, False),
            ("Salt & pepper", noisy, False),
            ("Median 3x3", median, False),
        ],
        "hw02_output.png",
    )
    print("Saved visualization to hw02_output.png")


if __name__ == "__main__":
    main()
