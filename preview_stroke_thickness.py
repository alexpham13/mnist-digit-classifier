"""Preview grayscale stroke thinning and thickening on one MNIST digit 5."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from sklearn.datasets import fetch_openml


def stroke_variants(pixels):
    """Blend 25% morphology with 75% original for gentle stroke changes."""
    # White ink has high values. Erosion takes the local minimum, shrinking
    # bright strokes; dilation takes the maximum, expanding them. These operate
    # on grayscale intensities directly, without converting to a binary image.
    thinner = grey_erosion(pixels, size=(3, 3), mode="constant", cval=0)
    thicker = grey_dilation(pixels, size=(3, 3), mode="constant", cval=0)
    # Convert to floating point before blending to avoid uint8 arithmetic
    # overflow, then round back to grayscale pixel values from 0 to 255.
    original_float = pixels.astype(np.float32)
    thinner = np.rint(0.75 * original_float + 0.25 * thinner).astype(np.uint8)
    thicker = np.rint(0.75 * original_float + 0.25 * thicker).astype(np.uint8)
    return thinner, pixels, thicker


def main():
    project_directory = Path(__file__).resolve().parent
    # Same MNIST version, parser, and local cache as explore_mnist.py.
    print("Loading MNIST from the project's data cache...", flush=True)
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff",
        data_home=str(project_directory / "data"),
    )
    training_labels = mnist.target[:60000].astype(np.int64)
    index = int(np.flatnonzero(training_labels == 5)[0])
    pixels = mnist.data[index].reshape(28, 28).astype(np.uint8)

    figure, axes = plt.subplots(1, 3, figsize=(10, 4))
    titles = ["Gently thinned\n75% original + 25% erosion", "Original strokes",
              "Gently thickened\n75% original + 25% dilation"]
    for axis, image, title in zip(axes, stroke_variants(pixels), titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255,
                    interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"MNIST training image #{index + 1}: digit 5, 3×3 kernel")
    figure.tight_layout()
    print("Preview ready: gently thinned, original, gently thickened. Close the plot window to exit.",
          flush=True)
    plt.show()


if __name__ == "__main__":
    main()
