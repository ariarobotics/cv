"""HW09: detect and track one moving animal using optical flow.

Submit only this file. Use the course cv Conda environment.
Allowed flow methods: cv2.calcOpticalFlowPyrLK,
cv2.calcOpticalFlowFarneback, or DISOpticalFlow_create(...).calc(...).
You may add helpers here. Keep experiments and file I/O out of the function.
"""
import cv2
import numpy as np


def track_animal(frames):
    """Return one [x, y, width, height] box for each input frame.

    frames: tuple of BGR uint8 arrays, each shaped (685, 1280, 3),
            in chronological order at 10 frames per second.
            Do not modify the supplied arrays; copy them if needed.
    returns: real numeric NumPy array of shape (len(frames), 4).
             Coordinates refer to the supplied image, not a resized image.
             Use [nan, nan, nan, nan] for a missing detection.

    No initial box is supplied. Use one algorithm and parameter policy for
    every clip. You may keep an estimate when the animal briefly stops.
    """
    boxes = np.full((len(frames), 4), np.nan, dtype=float)
    # TODO: detect the moving animal and track it using one allowed flow method.
    return boxes
