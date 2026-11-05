"""PROVIDED: photo loading and output checks for HW10. Leave this file unchanged."""
import importlib.util
import numbers
from pathlib import Path

import cv2
import numpy as np

PATTERN_SIZE = (8, 6)      # inner corners: (columns, rows)
MIN_PHOTOS, MAX_PHOTOS = 10, 20
MAX_UPLOAD_BYTES = 100_000_000
PHOTO_TYPES = {".jpg", ".jpeg", ".png"}


def load_module(path):
    """Import a student file (e.g., hw10.py) from its path."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("hw10_submission", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def square_size_mm(module):
    """Return SQUARE_SIZE_MM from the student's hw10.py."""
    value = getattr(module, "SQUARE_SIZE_MM", None)
    if value is None:
        raise ValueError("Set SQUARE_SIZE_MM at the top of hw10.py to the measured side length "
                         "of one checkerboard square, in mm.")
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not (
            np.isfinite(value) and value > 0):
        raise ValueError(f"SQUARE_SIZE_MM must be a positive number in mm, not {value!r}.")
    return float(value)


def read_image(path):
    """Read a JPEG/PNG photo as BGR uint8; also works with non-ASCII paths."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {Path(path).name}; save it as JPEG or PNG.")
    return image


def write_image(path, image):
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot write {path.name}")
    encoded.tofile(str(path))


class PhotoSequence:
    """A list of photos that reads each photo from disk only when it is used.

    It behaves like a list: len(images), images[i], and `for image in images:`.
    Full-resolution phone photos are large, so only one is in memory at a time.
    """

    def __init__(self, paths):
        self.paths = tuple(paths)
        self.size = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = read_image(path)
        size = (image.shape[1], image.shape[0])
        if self.size is None:
            self.size = size
        elif size != self.size:
            raise ValueError(f"{path.name} is {size[0]}x{size[1]} pixels, but earlier photos are "
                             f"{self.size[0]}x{self.size[1]}. All photos must have the same size "
                             "and orientation.")
        return image

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def load_photos(folder):
    """Return (paths, images) for the 10-20 JPEG/PNG photos in folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Photo folder not found: {folder}")
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in PHOTO_TYPES)
    if not MIN_PHOTOS <= len(paths) <= MAX_PHOTOS:
        raise ValueError(f"Found {len(paths)} JPEG/PNG photos in {folder}; provide 10-20.")
    total = sum(p.stat().st_size for p in paths)
    print(f"{len(paths)} photos, {total / 1e6:.1f} MB in total.")
    if total >= MAX_UPLOAD_BYTES:
        print("WARNING: Gradescope uploads must be smaller than 100 MB. Save your photos as "
              "JPEG with lower quality (keep the same width and height), or use fewer photos.")
    return paths, PhotoSequence(paths)


def check_projection_output(uv, n):
    """Check the shape and type returned by project_points."""
    if not isinstance(uv, np.ndarray):
        raise ValueError(f"project_points must return a NumPy array, not {type(uv).__name__}.")
    if uv.shape != (n, 2):
        raise ValueError(f"project_points must return shape ({n}, 2), not {uv.shape}.")
    if not np.issubdtype(uv.dtype, np.floating) or not np.isfinite(uv).all():
        raise ValueError("project_points must return finite floating-point coordinates.")


def check_calibration_output(output, n_detected):
    """Check the outputs of calibrate_camera; return rms, K, dist, rvecs, tvecs."""
    if not isinstance(output, (tuple, list)) or len(output) != 5:
        raise ValueError("calibrate_camera must return the five outputs of cv2.calibrateCamera: "
                         "rms, K, dist, rvecs, tvecs.")
    rms, K, dist, rvecs, tvecs = output
    K = np.asarray(K, dtype=float)
    dist = np.asarray(dist, dtype=float).ravel()
    rvecs = np.asarray(rvecs, dtype=float).reshape(-1, 3)
    tvecs = np.asarray(tvecs, dtype=float).reshape(-1, 3)
    if np.ndim(rms) != 0 or not np.isfinite(rms):
        raise ValueError("rms must be a single finite number.")
    if K.shape != (3, 3) or dist.size != 5:
        raise ValueError("K must be 3x3 and dist must have 5 values (k1, k2, p1, p2, k3).")
    if len(rvecs) != n_detected or len(tvecs) != n_detected:
        raise ValueError(f"Expected one board pose per detected photo ({n_detected}), "
                         f"got {len(rvecs)} rvecs and {len(tvecs)} tvecs.")
    if not all(np.isfinite(a).all() for a in (K, dist, rvecs, tvecs)):
        raise ValueError("Calibration outputs must be finite.")
    return float(rms), K, dist, rvecs, tvecs
