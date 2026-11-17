"""PROVIDED: loading, checks, and figures for HW11. Leave this file unchanged.

The autograder uses this same file, so `python evaluate.py` runs the same public
tests as Gradescope.
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

MAX_WIDTH = 1600          # larger photos are resized (and K scaled) before your code runs
PHOTO_TYPES = {".jpg", ".jpeg", ".png"}
POINTS = 0.4              # points per test, for each image pair
FUNCTIONS = {             # stage: (function name, number of returned values, names)
    "sparse": ("reconstruct_sparse", 5, "pts1, pts2, R, t, X"),
    "rectify": ("rectify_pair", 7, "rect1, rect2, R1, R2, P1, P2, Q"),
    "dense": ("reconstruct_dense", 3, "disparity, valid, points"),
}


# ---------------------------------------------------------------- loading

def load_module(path):
    """Import a student file (e.g., hw11.py) from its path."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("hw11_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_image(path):
    """Read a JPEG/PNG photo as BGR uint8, rotated upright as in HW10."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {Path(path).name}; save it as JPEG or PNG.")
    return image


def load_calibration(path, image):
    """Return K and dist from calibration.json; image_size must match the photo."""
    try:
        values = json.loads(Path(path).read_text())
        K = np.asarray(values["K"], dtype=float)
        dist = np.asarray(values["dist"], dtype=float).ravel()
        size = list(values["image_size"])
    except (OSError, ValueError, TypeError, KeyError):
        raise ValueError("Fill calibration.json with K, dist, and image_size from HW10.") from None
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or dist.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(dist).all()):
        raise ValueError("calibration.json: K must be a 3x3 camera matrix and dist must have 5 values.")
    if size != [image.shape[1], image.shape[0]]:
        raise ValueError(f"calibration.json has image_size {size}, but the photos are "
                         f"[{image.shape[1]}, {image.shape[0]}]. Use the same camera settings and "
                         "orientation as in HW10.")
    return K, dist


def shrink(image1, image2, K, max_width=MAX_WIDTH):
    """Resize both photos to at most max_width pixels wide and scale K to match.

    Lens distortion coefficients do not change with image size.
    """
    width = image1.shape[1]
    if width <= max_width:
        return image1, image2, K
    s = max_width / width
    size = (max_width, round(image1.shape[0] * s))
    sx, sy = size[0] / image1.shape[1], size[1] / image1.shape[0]
    K = K.copy()
    K[0, 0], K[0, 1], K[1, 1] = K[0, 0] * sx, K[0, 1] * sx, K[1, 1] * sy
    K[0, 2], K[1, 2] = (K[0, 2] + .5) * sx - .5, (K[1, 2] + .5) * sy - .5
    resize = lambda image: cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resize(image1), resize(image2), K


def photo_paths(folder):
    """The two photos in a folder: image1.* and image2.*, in that order."""
    paths = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in PHOTO_TYPES)
    if len(paths) != 2:
        raise ValueError(f"{folder} must contain exactly two JPEG/PNG photos, image1 and image2.")
    return paths


def load_pair(folder):
    """Return image1, image2, K, dist for a folder with two photos and calibration.json."""
    folder = Path(folder)
    image1, image2 = (read_image(p) for p in photo_paths(folder))
    if image1.shape != image2.shape:
        raise ValueError("The two photos must have the same width, height, and orientation.")
    K, dist = load_calibration(folder / "calibration.json", image1)
    image1, image2, K = shrink(image1, image2, K)
    return image1, image2, K, dist


def load_test_inputs(folder):
    """Reference pose and the inputs used to test rectify_pair and reconstruct_dense."""
    folder = Path(folder)
    reference = json.loads((folder / "reference_pose.json").read_text())
    with np.load(folder / "stage_inputs.npz", allow_pickle=False) as data:
        stage = {key: data[key] for key in data.files}
    stage["R_ref"] = np.asarray(reference["R_2_1"], dtype=float)
    stage["t_ref"] = np.asarray(reference["t_2_1"], dtype=float).ravel()
    return stage


def unpack(value, stage):
    """Check that a function returned the documented number of values; return NumPy arrays."""
    name, count, names = FUNCTIONS[stage]
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{name} must return {count} values: {names}.")
    arrays = []
    for item in value:
        array = np.asarray(item)
        if array.dtype.kind not in "biuf":
            raise ValueError(f"{name} must return NumPy arrays: {names}.")
        arrays.append(array)
    return arrays


# ---------------------------------------------------------------- observing OpenCV calls

def _sample(array):
    """A 24 x 24 grid of values from a 2D array, used to compare disparity maps."""
    a = np.asarray(array, dtype=float)
    if a.ndim != 2 or not a.size:
        return []
    rows = np.linspace(0, a.shape[0] - 1, min(24, a.shape[0])).astype(int)
    cols = np.linspace(0, a.shape[1] - 1, min(24, a.shape[1])).astype(int)
    return a[np.ix_(rows, cols)].ravel().tolist()


def _fixed_point(sample, raw_samples):
    """True if a disparity sample equals OpenCV's raw output, i.e., not divided by 16."""
    sample = np.asarray(sample, dtype=float)
    for raw, minimum in raw_samples:
        raw = np.asarray(raw, dtype=float)
        if raw.shape != sample.shape:
            continue
        valid = (raw > (minimum - 1) * 16) & (np.abs(raw) > 16)
        if valid.sum() >= 10 and np.mean(np.abs(sample[valid] - raw[valid]) < 1e-3) > .5:
            return True
    return False


class Observer:
    """Records which required OpenCV functions ran, and with which inputs."""

    def __init__(self):
        self.originals = {}
        self.reset()

    def reset(self):
        self.events = {"essential": 0, "pose": 0, "triangulate": 0, "rectify": [], "warp": 0,
                       "stereo": 0, "raw": [], "reproject": []}

    def _replace(self, name, function):
        self.originals[name] = getattr(cv2, name)
        setattr(cv2, name, function)

    def _count(self, name, event):
        original = getattr(cv2, name)

        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            self.events[event] += 1
            return result
        self._replace(name, wrapper)

    def install(self):
        """Call before importing the student file."""
        events = lambda: self.events   # reset() replaces the dictionary
        for name, event in [("recoverPose", "pose"), ("triangulatePoints", "triangulate"),
                            ("remap", "warp"), ("warpPerspective", "warp"), ("undistort", "warp")]:
            self._count(name, event)

        essential = cv2.findEssentialMat

        def find_essential(*args, **kwargs):
            result = essential(*args, **kwargs)
            method = kwargs.get("method", cv2.RANSAC)       # RANSAC is OpenCV's default
            if "method" not in kwargs:
                # Position of `method` in the three overloads: (p1, p2, K, method, ...),
                # (p1, p2, focal, pp, method, ...), (p1, p2, K1, d1, K2, d2, method, ...).
                if len(args) > 2 and np.shape(args[2]) == (3, 3):
                    index = 6 if len(args) > 4 and np.shape(args[4]) == (3, 3) else 3
                else:
                    index = 4
                if len(args) > index:
                    method = args[index]
            if np.isscalar(method) and method == cv2.RANSAC and result[0] is not None:
                events()["essential"] += 1
            return result
        self._replace("findEssentialMat", find_essential)

        stereo_rectify = cv2.stereoRectify

        def rectify(*args, **kwargs):
            result = stereo_rectify(*args, **kwargs)
            R = kwargs.get("R", args[5] if len(args) > 5 else None)
            T = kwargs.get("T", args[6] if len(args) > 6 else None)
            events()["rectify"].append([np.asarray(R, dtype=float).tolist(),
                                        np.asarray(T, dtype=float).ravel().tolist()])
            return result
        self._replace("stereoRectify", rectify)

        reproject_image = cv2.reprojectImageTo3D

        def reproject(*args, **kwargs):
            result = reproject_image(*args, **kwargs)
            disparity = kwargs.get("disparity", args[0] if args else None)
            Q = kwargs.get("Q", args[1] if len(args) > 1 else None)
            events()["reproject"].append({"Q": np.asarray(Q, dtype=float).tolist(),
                                          "unscaled": _fixed_point(_sample(disparity), events()["raw"])})
            return result
        self._replace("reprojectImageTo3D", reproject)

        class Matcher:
            """Wraps a StereoBM/StereoSGBM object to record its raw output."""
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def compute(self, *args, **kwargs):
                raw = self.inner.compute(*args, **kwargs)
                events()["stereo"] += 1
                events()["raw"].append([_sample(raw), self.inner.getMinDisparity()])
                return raw

        class Factory:
            """Wraps cv2.StereoBM / cv2.StereoSGBM so that .create() is observed too."""
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def __call__(self, *args, **kwargs):
                return Matcher(self.inner(*args, **kwargs))

            def create(self, *args, **kwargs):
                return Matcher(self.inner.create(*args, **kwargs))

        for name in ["StereoBM_create", "StereoSGBM_create"]:
            create = getattr(cv2, name)
            self._replace(name, lambda *a, _create=create, **k: Matcher(_create(*a, **k)))
        for name in ["StereoBM", "StereoSGBM"]:
            self._replace(name, Factory(getattr(cv2, name)))
        return self

    def restore(self):
        for name, original in self.originals.items():
            setattr(cv2, name, original)
        self.originals.clear()


# ---------------------------------------------------------------- output checks

def _finite(a, shape=None):
    return isinstance(a, np.ndarray) and a.dtype.kind in "biuf" and (shape is None or a.shape == shape) \
        and np.isfinite(a).all()


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def valid_pose(R, t):
    return (_finite(R, (3, 3)) and np.abs(R.T @ R - np.eye(3)).max() < 1e-3
            and abs(np.linalg.det(R) - 1) < 1e-3 and _finite(t) and t.size == 3 and np.linalg.norm(t) > 0)


def valid_sparse(outputs, shape):
    """Shapes and values of reconstruct_sparse's outputs; returns an error message or ''."""
    pts1, pts2, R, t, X = outputs
    h, w = shape[:2]
    if not valid_pose(R, t):
        return "R must be a 3x3 rotation matrix and t a nonzero 3-vector."
    if not (_finite(pts1) and pts1.ndim == 2 and pts1.shape[1] == 2 and len(pts1) >= 8
            and _finite(pts2, pts1.shape)):
        return "pts1 and pts2 must be N x 2 arrays of the same size, with N >= 8."
    if not all(((p >= 0) & (p < [w, h])).all() for p in (pts1, pts2)):
        return "pts1 and pts2 must be pixel coordinates inside the input images."
    if not _finite(X, (len(pts1), 3)):
        return "X must be an N x 3 array of finite points, one per match."
    return ""


def valid_rectified(outputs, shape):
    rect1, rect2, R1, R2, P1, P2, Q = outputs
    if not all(r.shape == shape and r.dtype == np.uint8 for r in (rect1, rect2)):
        return "rect1 and rect2 must be uint8 BGR images of the input size."
    if not all(_finite(a, s) for a, s in [(R1, (3, 3)), (R2, (3, 3)), (P1, (3, 4)), (P2, (3, 4)), (Q, (4, 4))]):
        return "R1, R2 must be 3x3, P1, P2 must be 3x4, and Q must be 4x4."
    return ""


def valid_dense(outputs, shape):
    disparity, valid, points = outputs
    h, w = shape[:2]
    if not (disparity.shape == (h, w) and disparity.dtype.kind == "f"):
        return "disparity must be a floating-point H x W array."
    if not (valid.shape == (h, w) and valid.dtype.kind in "biu"):
        return "valid must be an H x W Boolean array."
    mask = valid != 0
    if not mask.any():
        return "valid has no True pixels."
    if not (points.shape == (h, w, 3) and points.dtype.kind in "iuf"):
        return "points must be an H x W x 3 array."
    if not (np.isfinite(points[mask]).all() and (points[mask][:, 2] > 0).all()):
        return "Every valid pixel needs a finite 3D point with positive depth Z."
    return ""


def _result(name, passed, message, max_score=POINTS):
    return {"name": name, "score": max_score if passed else 0.0, "max_score": max_score,
            "status": "passed" if passed else "failed", "output": message}


def check_pose(outputs, trace, error, reference):
    name = "Relative pose"
    if outputs is None:
        return _result(name, False, error)
    R, t = outputs[2], outputs[3]
    if not valid_pose(R, t):
        return _result(name, False, "R must be a 3x3 rotation matrix and t a nonzero 3-vector.")
    if not (trace.get("essential") and trace.get("pose")):
        return _result(name, False, "Use cv2.findEssentialMat with RANSAC, then cv2.recoverPose.")
    cos = (np.trace(R @ reference["R_ref"].T) - 1) / 2
    rotation = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    direction = _angle(t.ravel().astype(float), reference["t_ref"])
    passed = rotation <= 5 and direction <= 20
    return _result(name, passed, f"Rotation error {rotation:.2f} deg (limit 5); "
                                 f"translation-direction error {direction:.2f} deg (limit 20).")


def check_triangulation(outputs, trace, error, shape, K, dist):
    name = "Triangulation"
    if outputs is None:
        return _result(name, False, error)
    message = valid_sparse(outputs, shape)
    if message:
        return _result(name, False, message)
    if not trace.get("triangulate"):
        return _result(name, False, "Use cv2.triangulatePoints.")
    pts1, pts2, R, t, X = outputs
    X = X.astype(float)
    X2 = X @ R.T + t.ravel()
    front = float(np.mean((X[:, 2] > 0) & (X2[:, 2] > 0)))
    errors = []
    for points, pixels in [(X, pts1), (X2, pts2)]:
        ideal = cv2.undistortPoints(pixels.reshape(-1, 1, 2).astype(float), K, dist, P=K).reshape(-1, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            projected = points @ K.T
            error_px = np.linalg.norm(projected[:, :2] / projected[:, 2:] - ideal, axis=1)
        errors.append(np.where(np.isfinite(error_px), error_px, np.inf))
    median = float(np.median(np.maximum(*errors)))
    passed = front >= .9 and median <= 3
    return _result(name, passed, f"{len(X)} points; {100 * front:.1f}% in front of both cameras (limit 90%); "
                                 f"median reprojection error {median:.2f} px (limit 3).")


def check_rectification(outputs, trace, error, shape, K, dist, stage):
    name = "Rectification"
    if outputs is None:
        return _result(name, False, error)
    message = valid_rectified(outputs, shape)
    if message:
        return _result(name, False, message)
    R_given, t_given = stage["R_2_1"], stage["t_2_1"].ravel()
    used = any(np.allclose(np.asarray(R), R_given, atol=1e-4)
               and np.allclose(np.asarray(T) / np.linalg.norm(T), t_given / np.linalg.norm(t_given), atol=1e-4)
               for R, T in trace.get("rectify", []) if np.shape(R) == (3, 3) and np.linalg.norm(T) > 0)
    if not used:
        return _result(name, False, "Call cv2.stereoRectify with the given R and t.")
    if not trace.get("warp"):
        return _result(name, False, "Warp both images, e.g., with cv2.initUndistortRectifyMap and cv2.remap.")
    _, _, R1, R2, P1, P2, _ = outputs
    a = cv2.undistortPoints(stage["points1"].reshape(-1, 1, 2), K, dist, R=R1, P=P1).reshape(-1, 2)
    b = cv2.undistortPoints(stage["points2"].reshape(-1, 1, 2), K, dist, R=R2, P=P2).reshape(-1, 2)
    gap = float(np.median(np.abs(a[:, 1] - b[:, 1])))
    return _result(name, gap <= 3, f"Median row difference of matches after rectification {gap:.2f} px (limit 3).")


def check_disparity(outputs, trace, error, shape):
    name = "Stereo disparity"
    if outputs is None:
        return _result(name, False, error)
    if not trace.get("stereo"):
        return _result(name, False, "Compute the disparity with cv2.StereoBM or cv2.StereoSGBM.")
    message = valid_dense(outputs, shape)
    if message:
        return _result(name, False, message)
    unscaled = _fixed_point(_sample(outputs[0]), trace.get("raw", [])) or \
        any(call["unscaled"] for call in trace.get("reproject", []))
    if unscaled:
        return _result(name, False, "Disparity must be in pixels: divide OpenCV's output by 16.")
    mask = outputs[1] != 0
    return _result(name, True, f"Disparity in pixels; {100 * mask.mean():.0f}% valid pixels (not graded).")


def check_points(outputs, trace, error, shape, Q):
    name = "Dense 3D points"
    if outputs is None:
        return _result(name, False, error)
    used = False
    for call in trace.get("reproject", []):
        q = np.asarray(call["Q"])
        if q.shape == (4, 4) and np.linalg.norm(q) > 0:
            a, b = q / np.linalg.norm(q), Q / np.linalg.norm(Q)
            used |= np.allclose(a, b, atol=1e-4) or np.allclose(a, -b, atol=1e-4)
    if not used:
        return _result(name, False, "Use cv2.reprojectImageTo3D with the given Q.")
    message = valid_dense(outputs, shape)
    if message:
        return _result(name, False, message)
    return _result(name, True, f"{int((outputs[1] != 0).sum())} valid 3D points.")


def run_tests(execute, pair, stage):
    """The five tests of one image pair. Each function is tested on its own inputs.

    execute(stage_name, arguments) returns (outputs or None, trace, error message).
    """
    image1, image2, K, dist = pair
    sparse = execute("sparse", (image1, image2, K, dist))
    rectified = execute("rectify", (image1, image2, K, dist, stage["R_2_1"], stage["t_2_1"]))
    dense = execute("dense", (stage["rect_image1"], stage["rect_image2"], stage["rect_Q"]))
    return [check_pose(*sparse, stage),
            check_triangulation(*sparse, image1.shape, K, dist),
            check_rectification(*rectified, image1.shape, K, dist, stage),
            check_disparity(*dense, stage["rect_image1"].shape),
            check_points(*dense, stage["rect_image1"].shape, stage["rect_Q"])]


def run_pipeline(execute, pair):
    """Run the three functions in sequence; return (all outputs, message). No accuracy checks."""
    image1, image2, K, dist = pair
    sparse, _, error = execute("sparse", (image1, image2, K, dist))
    if sparse is None or valid_sparse(sparse, image1.shape):
        return (sparse, None, None), "reconstruct_sparse: " + (error or valid_sparse(sparse, image1.shape))
    rectified, _, error = execute("rectify", (image1, image2, K, dist, sparse[2], sparse[3]))
    if rectified is None or valid_rectified(rectified, image1.shape):
        return (sparse, None, None), "rectify_pair: " + (error or valid_rectified(rectified, image1.shape))
    dense, _, error = execute("dense", (rectified[0], rectified[1], rectified[6]))
    if dense is None or valid_dense(dense, image1.shape):
        return (sparse, rectified, None), "reconstruct_dense: " + (error or valid_dense(dense, image1.shape))
    return (sparse, rectified, dense), ""


# ---------------------------------------------------------------- figure and point clouds

BLUE, RED, GREEN = "#2b6cb0", "#c53030", "#2f855a"


def save_ply(path, points, colors):
    """Colored point cloud for MeshLab, CloudCompare, or Open3D."""
    vertices = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, key in enumerate("xyz"):
        vertices[key] = points[:, i]
    for i, key in enumerate(["red", "green", "blue"]):
        vertices[key] = colors[:, i]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())


def _camera_top(ax, R, C, color, label, size, side):
    """A camera seen from above, as in the lecture: a triangle opening in its viewing direction."""
    corners = np.array([[-.55, 0, 1], [.55, 0, 1]]) * size @ R + C
    ax.fill([C[0], *corners[:, 0]], [C[2], *corners[:, 2]], color=color, alpha=.3, zorder=4)
    ax.plot([C[0], corners[0, 0], corners[1, 0], C[0]], [C[2], corners[0, 2], corners[1, 2], C[2]],
            color=color, linewidth=2, zorder=5)
    ax.plot(*corners[:, [0, 2]].T, color=color, linewidth=4, zorder=5)
    ax.text(C[0] + side * .8 * size, C[2], label, color=color, fontsize=14, fontweight="bold",
            ha="left" if side > 0 else "right", va="center", zorder=6)


def save_figure(path, image1, sparse, rectified, dense, title=""):
    """Matches colored by depth, rectified pair, disparity, and a top view of the reconstruction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=.1, wspace=.1)
    ax_pair = fig.add_subplot(grid[0, :])
    ax_matches, ax_disparity, ax_top = (fig.add_subplot(grid[1, k]) for k in range(3))
    for ax in (ax_pair, ax_matches, ax_disparity, ax_top):
        ax.axis("off")
    rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    clouds = []
    if sparse is not None:
        pts1, pts2, R, t, X = (np.asarray(a, dtype=float) for a in sparse)
        keep = np.isfinite(X).all(axis=1) & (X[:, 2] > 0)
        norm = plt.Normalize(*np.percentile(X[keep, 2], [2, 98])) if keep.any() else None
        ax_matches.imshow(rgb1)
        ax_matches.scatter(*pts1[keep].T, c=X[keep, 2], cmap="turbo", norm=norm, s=10,
                           edgecolors="black", linewidths=.3)
        ax_matches.set_title(f"Image 1: {keep.sum()} matches, colored by depth", fontsize=15)
        clouds.append(("sparse", X[keep], np.zeros((keep.sum(), 3)), R, t))
    if rectified is not None:
        pair = np.hstack([cv2.cvtColor(rectified[0], cv2.COLOR_BGR2RGB), cv2.cvtColor(rectified[1], cv2.COLOR_BGR2RGB)])
        ax_pair.imshow(pair)
        for y in np.linspace(.08, .92, 9) * pair.shape[0]:
            ax_pair.axhline(y, color="cyan", linewidth=.8)
        ax_pair.set_title("Rectified images 1 and 2: matching points lie on the same row", fontsize=15)
    if dense is not None and rectified is not None:
        disparity, valid, points = (np.asarray(a) for a in dense)
        mask = valid != 0
        cmap = plt.get_cmap("turbo").copy()
        cmap.set_bad("#1a202c")
        shown = ax_disparity.imshow(np.ma.masked_where(~mask, disparity), cmap=cmap)
        fig.colorbar(shown, ax=ax_disparity, orientation="horizontal", shrink=.9, pad=.03).set_label(
            "disparity (px); dark: invalid", fontsize=12)
        ax_disparity.set_title("Disparity of rectified image 1", fontsize=15)
        # points are in rectified camera 1; rotate them back into camera 1.
        X_dense = points[mask].astype(float) @ np.asarray(rectified[2], dtype=float)
        colors = cv2.cvtColor(rectified[0], cv2.COLOR_BGR2RGB)[mask]
        clouds.insert(0, ("dense", X_dense, colors, None, None))
    if clouds:
        ax = ax_top
        ax.axis("on")
        everything = np.vstack([c[1] for c in clouds])
        lo, hi = np.percentile(everything[:, [0, 2]], [2, 98], axis=0)
        for name, X, colors, _, _ in clouds:
            inside = ((X[:, [0, 2]] >= lo - .1 * (hi - lo)) & (X[:, [0, 2]] <= hi + .1 * (hi - lo))).all(axis=1)
            X, colors = X[inside], colors[inside]
            if name == "dense":
                step = max(1, len(X) // 40000)
                ax.scatter(X[::step, 0], X[::step, 2], c=colors[::step] / 255, s=1, linewidths=0)
            else:
                ax.scatter(X[:, 0], X[:, 2], c="black", s=4, linewidths=0)
        if sparse is not None:
            R, t = clouds[-1][3], clouds[-1][4].ravel()
            C2 = -R.T @ t                                 # camera 2's center in camera 1's frame
            size = min(.12 * max(hi - lo), .7 * np.linalg.norm(C2))
            side = 1 if C2[0] >= 0 else -1
            _camera_top(ax, np.eye(3), np.zeros(3), BLUE, "1", size, -side)
            _camera_top(ax, R, C2, GREEN, "2", size, side)
        ax.set_xlim(min(lo[0], -1) - .1 * (hi[0] - lo[0]), max(hi[0], 1) + .1 * (hi[0] - lo[0]))
        ax.set_ylim(-.12 * (hi[1] - lo[1]) - .5, hi[1] + .1 * (hi[1] - lo[1]))
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (baseline units)", fontsize=12)
        ax.set_ylabel("depth $z$", fontsize=12)
        ax.set_title("Top view: dense (color), sparse (black)", fontsize=15)
    if title:
        fig.suptitle(title, fontsize=17)
    fig.savefig(path, dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_results(folder, image1, sparse, rectified, dense, title=""):
    """Save reconstruction.jpg, sparse.ply, and dense.ply in folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_figure(folder / "reconstruction.jpg", image1, sparse, rectified, dense, title)
    if sparse is not None:
        pts1, X = np.asarray(sparse[0]), np.asarray(sparse[4], dtype=float)
        xy = np.clip(np.rint(pts1).astype(int), 0, [image1.shape[1] - 1, image1.shape[0] - 1])
        save_ply(folder / "sparse.ply", X, image1[xy[:, 1], xy[:, 0]][:, ::-1])
    if dense is not None and rectified is not None:
        mask = np.asarray(dense[1]) != 0
        X = np.asarray(dense[2], dtype=float)[mask] @ np.asarray(rectified[2], dtype=float)
        save_ply(folder / "dense.ply", X, rectified[0][mask][:, ::-1])
