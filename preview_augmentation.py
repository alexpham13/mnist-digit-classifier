"""Preview four simple MNIST augmentations without saving or training."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml


def make_previews(original):
    """Return labeled 28x28 grayscale images with black outside the digit."""
    # Pillow uses positive angles for counterclockwise rotation. Bilinear
    # interpolation softens rotated edges instead of making them jagged.
    clockwise = original.rotate(
        -10, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0
    )
    counterclockwise = original.rotate(
        10, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0
    )

    # Paste onto fresh black frames. Pixels that leave the frame are clipped,
    # never wrapped around to the opposite side. Integer shifts need no smoothing.
    left = Image.new("L", (28, 28), 0)
    left.paste(original, (-2, 0))
    right = Image.new("L", (28, 28), 0)
    right.paste(original, (2, 0))
    return [
        ("Original digit 5", original),
        ("10° clockwise", clockwise),
        ("10° counterclockwise", counterclockwise),
        ("Shift 2 pixels left", left),
        ("Shift 2 pixels right", right),
    ]


def main():
    # Reuse the dataset version, parser, and project cache from explore_mnist.py.
    project_directory = Path(__file__).resolve().parent
    print("Loading MNIST from the project's data cache...", flush=True)
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff",
        data_home=str(project_directory / "data"),
    )
    # Select only from the first 60,000 images: MNIST's training split.
    training_labels = mnist.target[:60000].astype(np.int64)
    index = int(np.flatnonzero(training_labels == 5)[0])
    pixels = mnist.data[index].reshape(28, 28).astype(np.uint8)
    original = Image.fromarray(pixels)

    figure, axes = plt.subplots(1, 5, figsize=(15, 4))
    for axis, (title, image) in zip(axes, make_previews(original)):
        # Nearest-neighbor display makes the resulting pixels easy to inspect;
        # rotation interpolation above is independent of display enlargement.
        axis.imshow(image, cmap="gray", vmin=0, vmax=255,
                    interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"MNIST training image #{index + 1}: augmentation preview")
    figure.tight_layout()
    print("Preview ready. View the Matplotlib window; close it to exit.", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
