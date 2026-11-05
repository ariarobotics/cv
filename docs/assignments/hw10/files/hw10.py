"""HW10: Camera simulation and calibration. Due Wednesday, November 11, 2026.

Set SQUARE_SIZE_MM and complete the four functions below.
Submit hw10.py and your photos.
Run:  python evaluate.py projection
      python evaluate.py calibration photos
"""
import cv2
import numpy as np

# TODO: side length of one square of YOUR printed or displayed checkerboard, in mm.
SQUARE_SIZE_MM = None


def project_points(X_w, K, R_c_w, t_c_w):
    """Return the pixel coordinates (u, v) of N world points, shape (N, 2).

    X_w:   (N, 3) array; row i is the world point ^wX_i.
    K:     (3, 3) intrinsic matrix.
    R_c_w: (3, 3) rotation ^cR_w from the world frame to the camera frame.
    t_c_w: (3, 1) translation ^ct_w from the world frame to the camera frame.

    Use NumPy. Do not call cv2.projectPoints in this function.
    """
    # TODO: world frame -> camera frame -> pixels (divide by the depth Z_c).
    raise NotImplementedError("Complete project_points")


def board_points(pattern_size, square_size):
    """Return the board points ^bX_i = (X, Y, 0), float32, shape (columns*rows, 3).

    pattern_size = (columns, rows) counts INNER corners, e.g., (8, 6).
    Start at (0, 0, 0) and go row by row: X increases fastest, then Y.
    Neighboring corners are square_size apart.
    """
    # TODO
    raise NotImplementedError("Complete board_points")


def detect_board(image, pattern_size):
    """Return the refined corners from cv2.cornerSubPix, or None if not found.

    image: BGR uint8 photo. Do not modify it.
    Use cv2.findChessboardCorners, then refine with cv2.cornerSubPix.
    Return the refined corners as OpenCV returns them: float32, one (x, y)
    per corner, columns*rows corners in OpenCV's order.
    """
    # TODO
    raise NotImplementedError("Complete detect_board")


def calibrate_camera(images, pattern_size, square_size):
    """Calibrate the camera from checkerboard photos.

    images: a list of BGR uint8 photos, all the same size.
            Loop over it with `for image in images:`.
    Skip photos in which detect_board returns None.

    Return the five outputs of cv2.calibrateCamera, unchanged:
        rms, K, dist, rvecs, tvecs
    """
    # TODO: use board_points, detect_board, and cv2.calibrateCamera.
    raise NotImplementedError("Complete calibrate_camera")
