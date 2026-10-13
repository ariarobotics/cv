"""Homework 07: Feature Matching.

Complete the three TODO functions. SIFT extraction, evaluation and plotting
are provided. Submit only hw07.py.
"""
from pathlib import Path
import argparse
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ratio_match(descriptors1, descriptors2, ratio):
    """Return forward cv2.DMatch objects passing the strict L2 ratio test.

    Descriptors are finite float32 arrays (N, D), (M, D), same D >= 1.
    Use BFMatcher(NORM_L2, crossCheck=False).knnMatch(..., k=2).
    Keep best if best.distance < ratio * second.distance; 0 < ratio < 1.
    N=0 or M<2: return []. Reject two zero-distance neighbors as ambiguous.
    Return matches in increasing queryIdx order; do not modify inputs.
    """
    # TODO: follow the corresponding assignment section.
    raise NotImplementedError("Complete ratio_match")


def mutual_matches(matches12, matches21):
    """Keep forward matches that also passed the reverse ratio test.

    Each input is a list of cv2.DMatch, with at most one match per query.
    For forward i->j, require reverse j->i. Preserve forward order and
    distances. Return a new list of the retained forward objects.
    Empty lists are valid. Do not modify either input or its match objects.
    """
    # TODO: follow the corresponding assignment section.
    raise NotImplementedError("Complete mutual_matches")


def match_images(image1, image2, ratio):
    """Return (keypoints1, keypoints2, forward_matches, mutual_matches).

    Inputs: 2-D uint8 grayscale images and 0 < ratio < 1. Use provided
    extract_sift(image) with its default nfeatures=800 for each image.
    Call ratio_match in both directions with ratio, then mutual_matches.
    Return the original keypoint lists, forward matches, and mutual subset.
    Do not change images. Empty feature sets are valid.
    """
    # TODO: follow the corresponding assignment section.
    raise NotImplementedError("Complete match_images")



# Everything below is PROVIDED. Leave it unchanged.
def extract_sift(image, nfeatures=800):
    """Return (keypoints, descriptors) from OpenCV SIFT, preserving order.

    image is 2-D uint8 grayscale. Create SIFT with nfeatures=nfeatures,
    leave other SIFT settings at defaults, then detectAndCompute(image, None).
    Return a list of cv2.KeyPoint and float32 descriptors of shape (N, 128).
    If no features: return [], np.empty((0, 128), dtype=np.float32).
    Do not change image; nfeatures is a positive integer.
    """
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return [], np.empty((0, 128), dtype=np.float32)
    return list(keypoints), descriptors


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def nearest_matches(descriptors1, descriptors2):
    if not len(descriptors1) or not len(descriptors2):
        return []
    return list(cv2.BFMatcher(cv2.NORM_L2).match(descriptors1, descriptors2))


def transformed_view(image):
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), 20, 0.80)
    result = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    return result, matrix


def evaluate_matches(keypoints1, keypoints2, matches, matrix, target_shape, tolerance=3.0):
    """Use a KNOWN affine transform; no transformation estimation is performed.

    Only evaluate matches whose source point maps inside the target view.
    Report positional consistency within tolerance, not a guarantee of identity.
    An empty evaluated set has no defined percentage (None).
    """
    if not matches:
        return {"accepted": 0, "evaluated": 0, "consistent": 0, "percent": None}
    source = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    target = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    projected = source @ matrix[:, :2].T + matrix[:, 2]
    h, w = target_shape[:2]
    visible = ((projected[:, 0] >= 0) & (projected[:, 0] <= w-1)
               & (projected[:, 1] >= 0) & (projected[:, 1] <= h-1))
    errors = np.linalg.norm(target-projected, axis=1)
    evaluated = int(visible.sum())
    consistent = int(((errors <= tolerance) & visible).sum())
    return {"accepted": len(matches), "evaluated": evaluated, "consistent": consistent,
            "percent": 100.0*consistent/evaluated if evaluated else None}


def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    ordered = sorted(matches, key=lambda m: (m.distance, m.queryIdx, m.trainIdx))
    indices = np.linspace(0, len(ordered)-1, min(40, len(ordered)), dtype=int)
    selected = [ordered[i] for i in indices]
    image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, selected, None,
                            matchColor=(20, 180, 255),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_demonstration(image1, image2, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keypoints1, keypoints2, forward, mutual = match_images(image1, image2, .75)
    _, d1 = extract_sift(image1)
    _, d2 = extract_sift(image2)
    raw = nearest_matches(d1, d2)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for ax, matches, label in zip(axes, [raw, forward, mutual],
                                 ["Nearest neighbor", "Ratio test (0.75)", "Ratio test in both directions + mutual"]):
        ax.imshow(draw_matches(image1, keypoints1, image2, keypoints2, matches))
        ax.set_title(f"{label}: {len(matches)} matches (showing up to 40)")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir/"hw07_matches.png", dpi=120)
    plt.close(fig)

    transformed, matrix = transformed_view(image1)
    kpa, da = extract_sift(image1)
    kpb, db = extract_sift(transformed)
    raw = nearest_matches(da, db)
    records = []
    def add_record(method, ratio, matches):
        metric = evaluate_matches(kpa, kpb, matches, matrix, transformed.shape)
        records.append({"method": method, "ratio": ratio, **metric})
        return metric
    add_record("nearest", "", raw)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for ax, threshold in zip(axes, [.60, .75, .90]):
        ka, kb, accepted, retained = match_images(image1, transformed, threshold)
        # All calls use identical provided SIFT settings and preserve its order.
        add_record("ratio", threshold, accepted)
        stats = add_record("mutual", threshold, retained)
        percentage = "n/a" if stats["percent"] is None else f'{stats["percent"]:.1f}%'
        ax.imshow(draw_matches(image1, ka, transformed, kb, retained))
        ax.set_title(f'Ratio {threshold:.2f}: {len(accepted)} forward, {len(retained)} mutual; '
                     f'{stats["consistent"]}/{stats["evaluated"]} evaluated consistent ({percentage})')
        ax.axis("off")
    fig.suptitle("Known rotation 20 degrees, scale 0.80; positional tolerance 3 pixels", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output_dir/"hw07_transformations.png", dpi=120)
    plt.close(fig)
    with (output_dir/"hw07_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "ratio", "accepted", "evaluated", "consistent", "percent"])
        writer.writeheader()
        writer.writerows(records)
    print("Known-transform experiment (percentage is among evaluated matches):")
    print("Method       Ratio  Accepted  Evaluated  Consistent  Percent")
    for row in records:
        percent = "n/a" if row["percent"] is None else f'{row["percent"]:.1f}%'
        print(f'{row["method"]:12} {str(row["ratio"]):5} {row["accepted"]:8} '
              f'{row["evaluated"]:10} {row["consistent"]:11} {percent:>8}')
    print("Saved hw07_matches.png, hw07_transformations.png, and hw07_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("image1", nargs="?", type=Path, default=base/"skyline_left.png")
    parser.add_argument("image2", nargs="?", type=Path, default=base/"skyline_right.png")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    save_demonstration(load_image(args.image1), load_image(args.image2), args.output_dir)


if __name__ == "__main__":
    main()
