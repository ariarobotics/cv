"""Homework 01: Working with Images.

Complete the TODO functions below. Do not change their names or arguments;
the Gradescope autograder calls these functions directly.

Run locally from the course `cv` Conda environment with:
    python hw01.py my_image.png

The program must print the requested values and save hw01_output.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_image(filename: str | Path) -> np.ndarray:
    """Load a color image from disk using OpenCV and return it."""
    # TODO: load the image with OpenCV and return the resulting NumPy array.
    raise NotImplementedError


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale float32 with values in [0, 1]."""
    # TODO: convert with OpenCV, then convert to np.float32 in the [0, 1] range.
    raise NotImplementedError


def top_left_3x5(gray_image: np.ndarray) -> np.ndarray:
    """Return the top-left block containing 3 rows and 5 columns."""
    # TODO
    raise NotImplementedError


def get_pixel_value(gray_image: np.ndarray, row: int, col: int) -> np.float32:
    """Return the pixel value at zero-based (row, col)."""
    # TODO
    raise NotImplementedError


def save_grayscale_image(
    gray_image: np.ndarray, output_path: str | Path = "hw01_output.png"
) -> None:
    """Save the grayscale image using OpenCV."""
    # TODO: save a viewable grayscale image with OpenCV.
    # Hint: cv2.imwrite expects conventional image intensities for a PNG.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="HW01: Working with Images")
    parser.add_argument("image", help="Path to your personal headshot image")
    args = parser.parse_args()

    image = load_image(args.image)
    gray = convert_to_grayscale(image)
    block = top_left_3x5(gray)
    pixel = get_pixel_value(gray, 1, 2)

    print("Top-left 3x5 grayscale block:")
    print(block)
    print("\nPixel value at row 1, column 2:")
    print(pixel)

    save_grayscale_image(gray, "hw01_output.png")
    print("\nSaved grayscale image to hw01_output.png")


if __name__ == "__main__":
    main()
