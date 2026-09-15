"""Homework 03: Laplacian Pyramids.

Complete the four TODO functions below. Do not change their names or
arguments; the Gradescope autograder calls them directly.

Run locally from the course `cv` Conda environment with:
    python hw03.py my_image.png

The script will save a visualization to hw03_output.png. You only submit
hw03.py to Gradescope.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build a Laplacian pyramid with `levels` residual levels.

    Return a list:
        [L0, L1, ..., L_(levels-1), G_levels]

    Requirements:
    - `image` is a 2-D np.float32 grayscale image with values in [0, 1].
    - Keep all pyramid arrays as np.float32.
    - For each level, use cv2.pyrDown(..., borderType=cv2.BORDER_REFLECT).
    - Expand the coarse image with cv2.pyrUp(..., dstsize=previous_shape[::-1]).
    - Compute each residual as fine_image - expanded_coarse_image.
    - The final list entry is the smallest Gaussian image.
    - IMPORTANT: Your implementation must work for odd and non-square image sizes.
    """
    # TODO
    raise NotImplementedError


def reconstruct_laplacian_pyramid(pyramid: list[np.ndarray]) -> np.ndarray:
    """Reconstruct the finest image from a Laplacian pyramid.

    Start with the final coarse image. Repeatedly expand it to the size of the
    residual at the next finer level and add that residual. Return np.float32.
    """
    # TODO
    raise NotImplementedError


def threshold_laplacian_pyramid(
    pyramid: list[np.ndarray], threshold: float
) -> list[np.ndarray]:
    """Return an independent thresholded copy of a Laplacian pyramid.

    For every residual level (all entries except the final coarse image), set
    coefficients whose absolute value is strictly less than `threshold` to 0.

    Return a new list containing a separate NumPy array for every level.
    Do NOT threshold the final coarse image, and do NOT modify or share arrays
    with the input pyramid. Keep all returned arrays as np.float32.

    A suitable independent copy for this list-of-arrays structure is:
        new_pyramid = [level.copy() for level in pyramid]
    """
    # TODO
    raise NotImplementedError


def residual_nonzero_fraction(pyramid: list[np.ndarray]) -> float:
    """Return the fraction of nonzero coefficients in the residual levels.

    Count coefficients only in pyramid[:-1]; the final coarse image is not
    included. Return a Python float in [0, 1].
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


def resize_for_pyramid(image: np.ndarray, max_dimension: int = 512) -> np.ndarray:
    """Downsize very large images to keep the visualization compact."""
    height, width = image.shape
    largest = max(height, width)
    if largest <= max_dimension:
        return image
    scale = max_dimension / largest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _to_display_u8(image: np.ndarray) -> np.ndarray:
    """Convert a reconstructed float image into a viewable uint8 image."""
    arr = np.asarray(image, dtype=np.float32)
    return np.clip(np.rint(np.clip(arr, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def save_comparison(
    panels: list[tuple[str, np.ndarray]],
    output_path: str | Path = "hw03_output.png",
) -> None:
    """Save a horizontal montage of reconstructions."""
    rendered = []
    for label, image in panels:
        panel = _to_display_u8(image)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 24), (255, 255, 255), -1)
        cv2.putText(
            panel,
            label,
            (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        rendered.append(panel)

    montage = cv2.hconcat(rendered)
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"Could not save {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HW03: Laplacian Pyramids and Compression")
    parser.add_argument("image", help="Path to your HW01 headshot or another JPG/PNG image")
    args = parser.parse_args()

    image = resize_for_pyramid(load_grayscale_float(args.image))
    pyramid = build_laplacian_pyramid(image, levels=3)

    exact = reconstruct_laplacian_pyramid(pyramid)
    max_error = float(np.max(np.abs(exact - image)))
    print("Exact reconstruction max error:", max_error)

    thresholds = [0.01, 0.03, 0.08]
    panels = [("Original", image), ("Exact", exact)]

    for threshold in thresholds:
        compressed = threshold_laplacian_pyramid(pyramid, threshold)
        fraction = residual_nonzero_fraction(compressed)
        reconstruction = reconstruct_laplacian_pyramid(compressed)
        print(f"Threshold {threshold:.2f} residual nonzero fraction: {fraction:.6f}")
        panels.append((f"T={threshold:.2f}", reconstruction))

    save_comparison(panels, "hw03_output.png")
    print("Saved visualization to hw03_output.png")


if __name__ == "__main__":
    main()
