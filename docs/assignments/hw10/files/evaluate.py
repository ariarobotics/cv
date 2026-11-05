"""PROVIDED local checks for HW10. Leave this file unchanged.

    python evaluate.py projection            # Part 1: public projection examples
    python evaluate.py calibration photos    # Part 2: run your code on your photos

Calibration results are saved in results/calibration/. Do not submit them.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from hw10_support import (PATTERN_SIZE, check_calibration_output, check_projection_output,
                          load_module, load_photos, square_size_mm, write_image)

HOME = Path(__file__).resolve().parent


# ---------------------------------------------------------------- Part 1

def rotation_z(degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def projection_examples():
    """Yield (name, X_w, K, R_c_w, t_c_w, expected uv)."""
    I, zero = np.eye(3), np.zeros((3, 1))
    K = np.array([[800., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    yield ("A: perspective division", np.array([[2., 4., 2.], [4., 8., 4.], [0., 0., 5.]]),
           I, I, zero, np.array([[1., 2.], [1., 2.], [0., 0.]]))
    yield ("B: intrinsics", np.array([[1., 2., 4.], [-1., 0., 2.]]),
           K, I, zero, np.array([[520., 540.], [-80., 240.]]))
    yield ("C: translation", np.array([[0., 0., 2.], [1., 2., 4.]]),
           I, I, np.array([[1.], [0.], [2.]]), np.array([[.25, 0.], [1 / 3, 1 / 3]]))
    yield ("D: rotation", np.array([[2., 0., 4.], [0., 2., 4.]]),
           I, rotation_z(90), zero, np.array([[0., .5], [-.5, 0.]]))
    R_y = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    yield ("E: rotation, translation, and intrinsics", np.array([[1., 2., 3.], [-1., 0., 2.]]),
           K, R_y, np.array([[.5], [-.5], [5.]]), np.array([[1020., 465.], [1960 / 3, 190.]]))
    # F: compare with OpenCV for 23 random points and a general camera pose.
    rng = np.random.default_rng(10)
    X_w = rng.uniform(-1., 1., size=(23, 3))
    rvec, t_c_w = np.array([.2, -.3, .1]), np.array([[.1], [-.2], [5.]])
    R_c_w, _ = cv2.Rodrigues(rvec)
    expected, _ = cv2.projectPoints(X_w, rvec, t_c_w, K, None)
    yield "F: 23 points vs. cv2.projectPoints", X_w, K, R_c_w, t_c_w, expected.reshape(-1, 2)


def check_projection(module):
    passed = total = 0
    for name, X_w, K, R_c_w, t_c_w, expected in projection_examples():
        total += 1
        try:
            uv = module.project_points(X_w.copy(), K.copy(), R_c_w.copy(), t_c_w.copy())
            check_projection_output(uv, len(X_w))
            np.testing.assert_allclose(uv, expected, atol=1e-4, rtol=0)
            print(f"PASS  {name}")
            passed += 1
        except NotImplementedError:
            print(f"FAIL  {name}: project_points is not implemented yet.")
        except AssertionError:
            print(f"FAIL  {name}:\n  expected\n{expected}\n  got\n{uv}")
        except Exception as error:
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
    print(f"{passed}/{total} public projection examples passed.")
    return passed == total


# ---------------------------------------------------------------- Part 2

def thumbnail(image, corners, label, ok, width=400):
    scale = width / image.shape[1]
    small = cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if corners is not None:
        cv2.drawChessboardCorners(small, PATTERN_SIZE, (corners * scale).astype(np.float32), True)
    cv2.rectangle(small, (0, 0), (width, 32), (255, 255, 255), -1)
    cv2.putText(small, label, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 140, 0) if ok else (0, 0, 220),
                2, cv2.LINE_AA)
    return small


def montage(tiles, columns=5):
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % columns)
    rows = [np.hstack([cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                       for t in tiles[i:i + columns]]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def calibration(module, folder, output):
    square_size = square_size_mm(module)
    print(f"SQUARE_SIZE_MM = {square_size:g} mm")
    paths, images = load_photos(folder)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Detect the board in every photo with YOUR detect_board.
    start = time.perf_counter()
    detections = []
    for path, image in zip(paths, images):
        corners = module.detect_board(image, PATTERN_SIZE)
        if corners is not None:
            corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
            if len(corners) != PATTERN_SIZE[0] * PATTERN_SIZE[1]:
                raise ValueError(f"detect_board returned {len(corners)} corners; expected 48.")
        detections.append(corners)
        print(f"{'FOUND    ' if corners is not None else 'NOT FOUND'}  {path.name}")
    found = [i for i, c in enumerate(detections) if c is not None]
    print(f"Board detected in {len(found)}/{len(paths)} photos ({time.perf_counter() - start:.0f} s). "
          "At least 10 are required.")
    if len(found) < 10:
        print("Retake the photos that were not found before submitting.")
    if not found:
        return

    # 2. Calibrate with YOUR calibrate_camera and check the outputs.
    try:
        output_values = module.calibrate_camera(images, PATTERN_SIZE, square_size)
    except NotImplementedError:
        print("calibrate_camera is not implemented yet.")
        output_values = None
    errors = {}
    if output_values is not None:
        rms, K, dist, rvecs, tvecs = check_calibration_output(output_values, len(found))
        X_b = np.asarray(module.board_points(PATTERN_SIZE, square_size), dtype=np.float64)
        print(f"\nRMS reprojection error: {rms:.3f} px")
        print("K =\n", np.array2string(K, precision=1, suppress_small=True))
        print("dist (k1, k2, p1, p2, k3) =", np.array2string(dist, precision=4))
        print("\nReprojection error of each view (rvecs[j], tvecs[j] -> j-th detected photo):")
        for j, i in enumerate(found):
            projected, _ = cv2.projectPoints(X_b, rvecs[j], tvecs[j], K, dist)
            residual = projected.reshape(-1, 2) - detections[i].reshape(-1, 2)
            errors[i] = np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
            print(f"  {paths[i].name:30s} {errors[i]:7.3f} px")

    # 3. Save an overview of all photos and an undistorted example.
    tiles = []
    for i, (path, image) in enumerate(zip(paths, images)):
        label = path.name + ("  NOT FOUND" if detections[i] is None else
                             f"  {errors[i]:.2f} px" if i in errors else "  FOUND")
        tiles.append(thumbnail(image, detections[i], label, detections[i] is not None))
        del image
    write_image(output / "overview.jpg", montage(tiles))
    if output_values is not None:
        image = images[found[0]]
        side = np.hstack([image, cv2.undistort(image, K, dist)])
        side = cv2.resize(side, (1600, round(side.shape[0] * 1600 / side.shape[1])), interpolation=cv2.INTER_AREA)
        for x, text in [(10, f"Original: {paths[found[0]].name}"), (810, "Undistorted with K and dist")]:
            cv2.rectangle(side, (x - 10, 0), (x + 790, 34), (255, 255, 255), -1)
            cv2.putText(side, text, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2, cv2.LINE_AA)
        write_image(output / "undistorted.jpg", side)
    print(f"\nSaved overview.jpg{' and undistorted.jpg' if output_values is not None else ''} in {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=("projection", "calibration"))
    parser.add_argument("photos", nargs="?", type=Path, help="folder with your 10-20 photos")
    parser.add_argument("--module", type=Path, default=HOME / "hw10.py")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        module = load_module(args.module)
        if args.part == "projection":
            if not check_projection(module):
                parser.exit(1)
        else:
            if args.photos is None:
                parser.error("give the photo folder: python evaluate.py calibration photos")
            calibration(module, args.photos, args.output_dir / "calibration")
    except NotImplementedError as error:
        parser.exit(1, f"Not implemented yet: {error}\n")
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
