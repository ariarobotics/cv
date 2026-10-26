"""Interface checks only; contains no detection or color-segmentation solution."""
import numpy as np


class ContractError(Exception):
    pass


def check_mask(mask, shape):
    if not isinstance(mask, np.ndarray) or mask.shape != shape:
        raise ContractError("red_mask must return a NumPy array with shape (H, W).")
    if mask.dtype != np.uint8 or not np.all((mask == 0) | (mask == 255)):
        raise ContractError("red_mask must return uint8 values 0 or 255.")


def check_circle(circle, shape):
    if circle is None:
        return
    if not isinstance(circle, np.ndarray) or circle.shape != (3,):
        raise ContractError("detect_sign must return a NumPy array of shape (3,), or None.")
    if circle.dtype.kind not in "iuf" or not np.isfinite(circle).all():
        raise ContractError("Circle coordinates must be finite real numbers.")
    x, y, r = circle.astype(float)
    h, w = shape
    if r <= 0 or x-r < 0 or y-r < 0 or x+r >= w or y+r >= h:
        raise ContractError("Return a positive-radius circle fully inside the image.")


"""Published localization bands; shared with the Gradescope grader."""

BANDS = ((0.10, 6.0, 1.0), (0.15, 9.0, 0.75),
         (0.25, 15.0, 0.50), (0.40, 24.0, 0.25))


def component_credit(error, reference_radius):
    for fraction, pixels, credit in BANDS:
        if error <= max(pixels, fraction * reference_radius):
            return credit
    return 0.0


def localization(prediction, reference):
    if prediction is None:
        return {"center_credit": 0.0, "radius_credit": 0.0,
                "center_error": None, "radius_error": None}
    p = np.asarray(prediction, dtype=float)
    ref = np.asarray(reference, dtype=float)
    center = float(np.linalg.norm(p[:2] - ref[:2]))
    radius = float(abs(p[2] - ref[2]))
    return {"center_credit": component_credit(center, ref[2]),
            "radius_credit": component_credit(radius, ref[2]),
            "center_error": center, "radius_error": radius}
