"""HW08: Document corners. Due Wednesday, October 28, 2026.

Complete detect_document. Keep intersect_lines unchanged.
Submit only hw08.py. Run: python evaluate.py
"""
import cv2
import numpy as np


def intersect_lines(line1, line2):
    """Intersect two Hough (r, theta) lines; return (x,y), or None if parallel."""
    r1, t1 = line1
    r2, t2 = line2
    matrix = np.array([[np.cos(t1), np.sin(t1)],
                       [np.cos(t2), np.sin(t2)]], dtype=float)
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    return np.linalg.solve(matrix, np.array([r1, r2], dtype=float))


def detect_document(image):
    """Return a numeric NumPy array of four (x, y) corners, shape (4, 2).

    Input: BGR uint8 image, shape (1000, 750, 3). Do not modify it.
    Corner order is arbitrary; use original input coordinates.
    Use cv2.HoughLines and the provided intersect_lines helper.
    Return np.empty((0, 2)) if no document can be detected.
    Additional helper functions are allowed in this file.
    """
    raise NotImplementedError("Complete detect_document.")
