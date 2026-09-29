"""Homework 05: Edge Detection with Canny — student starter.

Complete the four TODO functions. All helpers and plotting are provided.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gaussian_derivative_kernels(sigma):
    """PROVIDED: return float32 x/y derivative filters for cv2.filter2D."""
    radius = int(np.ceil(4 * sigma))
    g = cv2.getGaussianKernel(2 * radius + 1, sigma).astype(np.float32)
    gaussian = g @ g.T
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    x, y = np.meshgrid(coordinates, coordinates)
    # Positive derivative for intensity increasing toward right/bottom.
    hx = (x / sigma**2) * gaussian
    hy = (y / sigma**2) * gaussian
    return hx.astype(np.float32), hy.astype(np.float32)


def sample_along_gradient(magnitude, angle):
    """PROVIDED: sample one pixel forward/backward along angle (radians).

    Columns are x; rows are y. OpenCV performs the interpolation.
    """
    y, x = np.indices(magnitude.shape, dtype=np.float32)
    dx = np.cos(angle).astype(np.float32)
    dy = np.sin(angle).astype(np.float32)
    forward = cv2.remap(
        magnitude, x + dx, y + dy, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE)
    backward = cv2.remap(
        magnitude, x - dx, y - dy, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE)
    return forward, backward


def compute_gradients(image, sigma):
    """Return (Ix, Iy, magnitude, angle), all same-size float32 arrays.

    image: 2-D float32 grayscale in [0,1]; sigma: positive float.
    Use the provided kernels and cv2.filter2D with BORDER_REFLECT_101.
    Keep signed derivatives; angle is np.arctan2(Iy, Ix), in radians.
    """
    # TODO: follow Part 1 in the assignment.
    raise NotImplementedError("Complete compute_gradients")


def nonmaximum_suppression(magnitude, angle):
    """Return same-size float32 magnitudes after thinning.

    Use sample_along_gradient; keep M > forward AND M >= backward.
    Suppress other pixels to zero and zero the outermost border.
    Do not modify either input.
    """
    # TODO: follow Part 2 in the assignment.
    raise NotImplementedError("Complete nonmaximum_suppression")


def hysteresis_threshold(response, low, high):
    """Return a same-size bool edge map using 8-neighbor connectivity.

    Candidates: response >= low AND response > 0; strong: candidates
    with response >= high. Keep candidate components with a strong pixel.
    Valid thresholds satisfy 0 <= low <= high. Do not change response.
    """
    # TODO: follow Part 3 in the assignment.
    raise NotImplementedError("Complete hysteresis_threshold")


def detect_edges(image, sigma, low, high):
    """Call your three functions in order and return the final bool edge map."""
    # TODO: follow Part 4 in the assignment.
    raise NotImplementedError("Complete detect_edges")


# Everything below is PROVIDED. Leave it unchanged.
def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return image.astype(np.float32) / 255.0


def save_demonstration(image, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma, low, high = 2.0, 0.008, 0.020
    _, _, magnitude, angle = compute_gradients(image, sigma)
    response = nonmaximum_suppression(magnitude, angle)
    edges = detect_edges(image, sigma, low, high)
    cv2.imwrite(str(output_dir / 'hw05_edges.png'), edges.astype(np.uint8) * 255)

    limit = max(float(magnitude.max()), 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    panels = [(image, 'Input image', 1),
              (magnitude, 'Gradient magnitude', limit),
              (response, 'After non-maximum suppression', limit),
              (edges, 'Hysteresis: final edges', 1)]
    for ax, (values, title, vmax) in zip(axes.flat, panels):
        ax.imshow(values, cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=15)
        ax.axis('off')
    fig.tight_layout(pad=1.2)
    fig.savefig(output_dir / 'hw05_output.png', dpi=130)
    plt.close(fig)

    settings = [(1.0, 0.008, 0.020), (2.0, 0.008, 0.020),
                (4.0, 0.008, 0.020), (2.0, 0.016, 0.040)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (s, lo, hi) in zip(axes.flat, settings):
        result = detect_edges(image, s, lo, hi)
        ax.imshow(result, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'σ = {s:g}, low = {lo:g}, high = {hi:g}', fontsize=15)
        ax.axis('off')
    fig.tight_layout(pad=1.2)
    fig.savefig(output_dir / 'hw05_parameters.png', dpi=130)
    plt.close(fig)
    print('Saved hw05_edges.png, hw05_output.png, and hw05_parameters.png')


def main():
    parser = argparse.ArgumentParser(description='Homework 05: Edge Detection with Canny')
    parser.add_argument('image', nargs='?', type=Path,
                        default=Path(__file__).resolve().parent / 'dashcam.jpg')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    save_demonstration(load_image(args.image), args.output_dir)


if __name__ == '__main__':
    main()
