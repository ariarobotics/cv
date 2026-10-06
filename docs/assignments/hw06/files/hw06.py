"""Homework 06: Feature Point Detection.

Complete the two TODO functions. Helpers and demonstration are provided.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def derivative_filters():
    """PROVIDED: normalized Sobel filters; positive x right, positive y down."""
    hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8
    return hx, hx.T.copy()


def harris_response(image, sigma_d, sigma_w, alpha):
    """Return same-size float32 Harris scores without modifying image.

    image is 2-D float32 in [0, 1]. Smooth with sigma_d, apply the
    provided derivative_filters() using cv2.filter2D, smooth the three
    gradient products with sigma_w, and compute det(M)-alpha*trace(M)**2.
    Use BORDER_REFLECT_101 and GaussianBlur(..., (0, 0), sigmaX=sigma).
    Keep signed responses. All parameters are valid; both sigmas > 0.
    """
    # TODO: follow the corresponding assignment section.
    raise NotImplementedError("Complete harris_response")


def extract_sift(image, nfeatures=800):
    """Return (keypoints, descriptors) from OpenCV SIFT, preserving order.

    image is 2-D uint8 grayscale. Create SIFT with nfeatures=nfeatures,
    leave other SIFT settings at defaults, then detectAndCompute(image, None).
    Return a list of cv2.KeyPoint and float32 descriptors of shape (N, 128).
    If no features: return [], np.empty((0, 128), dtype=np.float32).
    Do not change image; nfeatures is a positive integer.
    """
    # TODO: follow the corresponding assignment section.
    raise NotImplementedError("Complete extract_sift")


# Everything below is PROVIDED. Leave it unchanged.
def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def select_corners(response):
    """Select positive local maxima. This is provided, not a student task."""
    maxima = cv2.dilate(response, np.ones((13, 13), dtype=np.uint8))
    keep = (response > 0) & (response > 0.01 * max(0., float(response.max())))
    keep &= response == maxima
    keep[:7] = False
    keep[-7:] = False
    keep[:, :7] = False
    keep[:, -7:] = False
    y, x = np.nonzero(keep)
    return x, y


def show_harris(ax, image, response):
    x, y = select_corners(response)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.scatter(x, y, s=20, facecolors="none", edgecolors="#ffb000", linewidths=1)
    ax.set_title(f"Harris: {len(x)} selected corners")
    ax.axis("off")


def show_sift(ax, image, keypoints):
    shown = sorted(keypoints, key=lambda k: -k.response)[:120]
    view = cv2.drawKeypoints(image, shown, None, color=(20, 180, 255),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
    ax.set_title(f"SIFT: {len(keypoints)} keypoints (showing up to 120)")
    ax.axis("off")


def transformed_views(image):
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), 20, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    smaller = cv2.resize(image, None, fx=0.65, fy=0.65, interpolation=cv2.INTER_AREA)
    return [("Original", image), ("Rotated 20 degrees", rotated), ("Scaled to 65%", smaller)]


def save_demonstration(image, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    response = harris_response(image.astype(np.float32)/255, 1.0, 2.0, 0.04)
    keypoints, descriptors = extract_sift(image, nfeatures=800)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    show_harris(axes[0], image, response)
    limit = max(float(np.abs(response).max()), 1e-12)
    axes[1].imshow(response, cmap="gray", vmin=-limit, vmax=limit)
    axes[1].set_title("Harris score: dark < 0, light > 0")
    axes[1].axis("off")
    show_sift(axes[2], image, keypoints)
    fig.tight_layout()
    fig.savefig(output_dir/"hw06_features.png", dpi=140)
    plt.close(fig)

    views = transformed_views(image)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for row, (label, view) in enumerate(views):
        scores = harris_response(view.astype(np.float32)/255, 1.0, 2.0, 0.04)
        kp, _ = extract_sift(view, nfeatures=800)
        show_harris(axes[row, 0], view, scores)
        show_sift(axes[row, 1], view, kp)
        for ax in axes[row]:
            ax.set_title(label + "\n" + ax.get_title())
    fig.tight_layout()
    fig.savefig(output_dir/"hw06_transformations.png", dpi=130)
    plt.close(fig)
    print(f"SIFT: {len(keypoints)} keypoints; descriptors shape: {descriptors.shape}")
    print("Saved hw06_features.png and hw06_transformations.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent/"skyline.png")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    save_demonstration(load_image(args.image), args.output_dir)


if __name__ == "__main__":
    main()
