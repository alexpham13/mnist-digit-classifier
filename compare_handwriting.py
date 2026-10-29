"""Compare a saved handwritten 5 with five MNIST test-set examples."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml


PROJECT_DIRECTORY = Path(__file__).resolve().parent
EXAMPLES_DIRECTORY = PROJECT_DIRECTORY / "collected_examples"


def load_saved_five(metadata_path=None):
    """Read an actual-digit label and its already-preprocessed grayscale PNG."""
    paths = [metadata_path] if metadata_path else sorted(
        EXAMPLES_DIRECTORY.glob("*.json"), reverse=True
    )
    for path in paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("actual_digit") != 5:
            if metadata_path:
                raise ValueError("The selected example must have actual_digit 5.")
            continue
        filename = metadata.get("image_file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Invalid image_file in {path}")
        image_path = path.parent / filename
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (28, 28):
                raise ValueError("Saved example must be a 28×28 grayscale PNG.")
            # PNGs already contain the app's cropped, resized, centered input.
            # Display these exact pixels without preprocessing them again.
            return np.array(image), image_path
    raise ValueError("No saved examples labeled as digit 5 were found.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, help="JSON metadata for a saved 5")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for repeatable MNIST selection (default: 42)")
    arguments = parser.parse_args()
    try:
        saved_pixels, saved_path = load_saved_five(arguments.example)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Cannot load saved handwriting: {error}") from error

    # Match explore_mnist.py: same dataset version, parser, and local cache.
    print("Loading MNIST from the project's data cache...", flush=True)
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff",
        data_home=str(PROJECT_DIRECTORY / "data"),
    )
    # The last 10,000 images are MNIST's standard test set.
    test_images = mnist.data[60000:]
    test_labels = mnist.target[60000:].astype(np.int64)
    five_indices = np.flatnonzero(test_labels == 5)
    rng = np.random.default_rng(arguments.seed)
    selected = rng.choice(five_indices, size=5, replace=False)

    figure, axes = plt.subplots(2, 3, figsize=(11, 8))
    panels = [(saved_pixels, "Your saved handwriting\nActual digit: 5")]
    for index in selected:
        panels.append((test_images[index].reshape(28, 28),
                       f"MNIST test image #{index + 1}\nActual digit: 5"))
    for axis, (pixels, title) in zip(axes.flat, panels):
        # Nearest-neighbor enlargement shows square pixels without smoothing.
        # A shared brightness range makes the comparison fair across all panels.
        axis.imshow(pixels, cmap="gray", vmin=0, vmax=255,
                    interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Your handwritten 5 compared with MNIST test-set 5s", fontsize=16)
    figure.text(0.5, 0.02, f"Saved example: {saved_path.name}",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    print(f"Saved example: {saved_path}")
    print("MNIST test image numbers (1-based):", (selected + 1).tolist())
    print("View the six enlarged images in the Matplotlib window. Close it to exit.",
          flush=True)
    plt.show()


if __name__ == "__main__":
    main()
