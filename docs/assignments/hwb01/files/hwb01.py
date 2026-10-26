"""Bonus homework: color segmentation and circular speed-sign detection.

Complete both functions. Submit this file only. Keep any additional helpers
in this file. Do not run experiments, load files, or display figures on import.
"""
import cv2
import numpy as np


def red_mask(image):
    """Return a uint8 mask of red pixels, shape (H, W), values 0 or 255.

    Input: BGR uint8 image. Do not modify it. Implement the color segmentation
    yourself, using Lecture 09. No supplied red-mask implementation is used.
    """
    raise NotImplementedError("Implement red_mask.")


def detect_sign(image):
    """Return a numeric NumPy array [center_x, center_y, radius], shape (3,).

    Input: BGR uint8 image, shape (1000, 750, 3). Do not modify it.
    Use your red_mask result and cv2.HoughCircles in the detection pipeline.
    Return original image coordinates. Target the outside of the red border.
    Return None if no circle is detected. Each graded photo contains one target.
    Additional helper functions are allowed in this file.
    """
    raise NotImplementedError("Implement detect_sign.")
