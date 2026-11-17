"""HW11: 3D reconstruction from two views. Due Monday, November 23, 2026.

Complete the three functions below. Submit hw11.py, your two photos, and calibration.json.
Run:  python evaluate.py
      python evaluate.py --photos photos
"""
import cv2
import numpy as np


def reconstruct_sparse(image1, image2, K, dist):
    """Relative pose and sparse 3D points from two photos.

    image1, image2: BGR uint8 photos of the same size.
    K: (3, 3) intrinsic matrix.  dist: lens distortion coefficients.

    Return pts1, pts2, R, t, X:
        pts1, pts2: (N, 2) matched pixels in image1 and image2 (as given, distorted)
        R: (3, 3) and t: (3, 1) pose of camera 2 relative to camera 1, X_2 = R X_1 + t
        X: (N, 3) triangulated points in camera 1's frame; row i belongs to match i
    """
    # TODO: match features, undistort the matches, cv2.findEssentialMat (RANSAC),
    #       cv2.recoverPose, cv2.triangulatePoints.
    raise NotImplementedError("Complete reconstruct_sparse")


def rectify_pair(image1, image2, K, dist, R, t):
    """Rectify both photos using the pose R, t of camera 2 relative to camera 1.

    Return rect1, rect2, R1, R2, P1, P2, Q:
        rect1, rect2: rectified BGR uint8 images, the same size as the inputs
        R1, R2, P1, P2, Q: the outputs of cv2.stereoRectify
    """
    # TODO: cv2.stereoRectify, cv2.initUndistortRectifyMap, cv2.remap.
    raise NotImplementedError("Complete rectify_pair")


def reconstruct_dense(rect1, rect2, Q):
    """Disparity and dense 3D points from a rectified pair.

    Return disparity, valid, points:
        disparity: (H, W) float32 disparity in pixels
        valid: (H, W) Boolean, True where the 3D point is usable (finite, positive depth)
        points: (H, W, 3) 3D points in rectified camera 1's frame
    """
    # TODO: cv2.StereoBM or cv2.StereoSGBM, divide by 16, cv2.reprojectImageTo3D.
    raise NotImplementedError("Complete reconstruct_dense")
