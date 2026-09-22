"""HW04: Hybrid Images with Fourier Transform.
Complete only the three marked functions. Use the course cv environment.
Do not use generative AI to solve this homework.
"""
from pathlib import Path
import argparse

import cv2
import numpy as np


def log_magnitude_spectrum(image):
    """Return centered log(1 + |DFT(image)|), float32, same shape.

    image: 2-D float32 array (may contain negative values).
    Use cv2.dft with complex output. Do not scale the forward DFT.
    Shift only axes (0, 1); do not shift the real/imaginary channel axis.
    Do not mutate image or normalize the returned spectrum.
    """
    # 1. cv2.dft(..., flags=cv2.DFT_COMPLEX_OUTPUT).
    # 2. Center axes (0, 1) with np.roll; shifts are (M//2, N//2).
    # 3. cv2.magnitude(F[..., 0], F[..., 1]), then np.log1p.
    # TODO: return the centered log-magnitude image.
    raise NotImplementedError("Complete log_magnitude_spectrum")


def gaussian_frequency_mask(shape, sigma):
    """PROVIDED: return a centered Gaussian low-pass mask. Leave unchanged.

    Call gaussian_frequency_mask(image.shape, sigma).
    Larger sigma gives stronger smoothing. For high-pass use 1 - mask.
    You do not need to implement or derive this helper.
    """
    rows, cols = shape
    u = (np.arange(cols, dtype=np.float64) - cols // 2) / cols
    v = (np.arange(rows, dtype=np.float64) - rows // 2) / rows
    U, V = np.meshgrid(u, v)
    return np.exp(-2*np.pi**2*sigma**2*(U**2+V**2)).astype(np.float32)


def apply_frequency_filter(image, mask):
    """Filter using cv2.dft/cv2.idft; return a same-shape float32 array.

    image: 2-D float32 array. mask: centered 2-D float32 frequency weights.
    Graded masks preserve conjugate symmetry. Use periodic boundaries,
    without padding. Preserve signs and phase. Do not clip, normalize,
    take an absolute value, or modify either input.
    """
    # 1. Compute the complex DFT, then center axes (0, 1).
    # 2. Multiply by mask[..., None] to weight BOTH complex channels.
    # 3. Undo the exact shifts: (-(M//2), -(N//2)).
    # 4. cv2.idft(..., flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT).
    # TODO: return the filtered float32 image without clipping.
    raise NotImplementedError("Complete apply_frequency_filter")


def make_hybrid_image(image_low, image_high, sigma_low, sigma_high):
    """Return (low, high, hybrid), each a same-shape float32 array.

    Both inputs have the same shape and values in [0, 1].
    Use gaussian_frequency_mask and apply_frequency_filter.
    low = low-pass(image_low, sigma_low)
    high = high-pass(image_high, sigma_high), with mask 1 - H_high
    hybrid = (low + high) / 2
    Do not clip, normalize, or modify inputs.
    """
    # 1. Use PROVIDED gaussian_frequency_mask(image_low.shape, sigma_low).
    # 2. Pass image_low and that mask to YOUR apply_frequency_filter.
    # 3. Make a second Gaussian mask for image_high and sigma_high.
    # 4. Pass image_high and (1 - second_mask) to YOUR filter function.
    # 5. Average the two resulting images and return (low, high, hybrid).
    # TODO: connect the provided helper and your filter function.
    raise NotImplementedError("Complete make_hybrid_image")


# Everything below is provided. Keep its output contract unchanged.
def load_grayscale(path):
    """Use the exact supplied pixels: no cropping, resizing, or alignment."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image.astype(np.float32) / np.float32(255)


def to_uint8(image):
    """Fixed conversion for saving; this does not modify the float image."""
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def save_comparison(images, spectra, out_path):
    # Matplotlib is used only for display; OpenCV handles image operations.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ["Marilyn | low-frequency source", "Einstein | high-frequency source",
              "Marilyn | low-pass, sigma = 9", "Einstein | high-pass, sigma = 2",
              "Hybrid | (low + high) / 2"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.7), layout="constrained")
    fig.suptitle("Homework 04: Hybrid Images with Fourier Transform", fontsize=20,
                 fontweight="bold", color="#17324D")
    upper = max(1.0, max(float(s.max()) for s in spectra))
    for j, (im, spec, title) in enumerate(zip(images, spectra, titles)):
        # Display only: zero high-pass response is mid-gray. Double the
        # averaged hybrid for display to restore the low-pass brightness.
        display = np.clip(im + 0.5, 0, 1) if j == 3 else np.clip(2*im, 0, 1) if j == 4 else im
        axes[0, j].imshow(display, cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(title, fontsize=10, pad=10)
        axes[1, j].imshow(spec, cmap="gray", vmin=0, vmax=upper)
        axes[1, j].set_title("Centered log-magnitude spectrum", fontsize=9, pad=10)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.supxlabel("High-pass display: clip(high + 0.5)  |  Hybrid display: clip(2 x hybrid)  |  Spectra share one scale", fontsize=10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_distance_view(hybrid, out_path):
    """A progressively smaller view; display gain is fixed at two."""
    display = to_uint8(2 * hybrid)
    h, w = display.shape
    gap = 20
    panels = [display]
    for divisor in (2, 4, 8):
        panels.append(cv2.resize(display, (max(1, w//divisor), max(1, h//divisor)),
                                 interpolation=cv2.INTER_AREA))
    canvas = np.full((h + 2*gap, sum(p.shape[1] for p in panels) + 5*gap), 235, np.uint8)
    x = gap
    for panel in panels:
        y = gap + (h-panel.shape[0])//2
        canvas[y:y+panel.shape[0], x:x+panel.shape[1]] = panel
        x += panel.shape[1] + gap
    if not cv2.imwrite(str(out_path), canvas):
        raise OSError(f"Cannot save {out_path}")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("marilyn", nargs="?", type=Path, default=here/"marilyn.png")
    parser.add_argument("einstein", nargs="?", type=Path, default=here/"einstein.png")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    a, b = load_grayscale(args.marilyn), load_grayscale(args.einstein)
    if a.shape != b.shape:
        raise ValueError("Inputs must have matching dimensions; do not resize the supplied images.")
    low, high, hybrid = make_hybrid_image(a, b, 9.0, 2.0)
    images = [a, b, low, high, hybrid]
    spectra = [log_magnitude_spectrum(im) for im in images]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_dir/"hw04_hybrid.png"), to_uint8(hybrid)):
        raise OSError("Could not save hw04_hybrid.png")
    save_comparison(images, spectra, args.output_dir/"hw04_output.png")
    save_distance_view(hybrid, args.output_dir/"hw04_distance.png")
    print(f"Input shape: {a.shape}")
    print("Parameters: sigma_low=9, sigma_high=2; hybrid=(low+high)/2")
    print(f"High-pass mean (approximately zero): {float(high.mean()):.8f}")
    print("Saved hw04_hybrid.png, hw04_output.png, hw04_distance.png")


if __name__ == "__main__":
    main()
