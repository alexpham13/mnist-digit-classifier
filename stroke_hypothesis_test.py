"""Test one fixed local stroke edit without altering the saved example."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


PROJECT_DIRECTORY = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_DIRECTORY / "collected_examples" / (
    "20261028T165145_a49c1ed8fff5415fbf9aacb9f152f8b8.png"
)
MODEL_PATH = PROJECT_DIRECTORY / "models" / "mnist_svm.joblib"


def normalize(pixels):
    # Saved PNGs already have the app's crop/resize/centering. Repeating those
    # operations would change other regions and invalidate this local test.
    return (pixels.astype(np.float32) / 255.0).reshape(1, 784)


def strengthen_stroke(original):
    """Brighten only the approved 4x3 region, with no changes elsewhere."""
    mask = np.zeros((28, 28), dtype=bool)
    mask[5:9, 12:15] = True  # Inclusive rows 5–8 and columns 12–14.
    modified = original.copy()
    # Wider integers prevent overflow when adding 40 to near-white pixels.
    modified[mask] = np.minimum(original[mask].astype(np.int16) + 40, 255)
    assert np.array_equal(modified[~mask], original[~mask])
    assert np.all(modified[mask] >= original[mask])
    difference = modified.astype(np.int16) - original.astype(np.int16)
    assert not np.any(difference[~mask])
    return modified, difference


def main():
    metadata = json.loads(IMAGE_PATH.with_suffix(".json").read_text())
    if type(metadata.get("actual_digit")) is not int or metadata["actual_digit"] != 5:
        raise SystemExit(f"Metadata mismatch: expected actual_digit 5, got {metadata.get('actual_digit')!r}")
    with Image.open(IMAGE_PATH) as image:
        if image.mode != "L" or image.size != (28, 28) or image.format != "PNG":
            raise SystemExit("Image mismatch: expected a 28×28 grayscale PNG.")
        original = np.array(image)
    model = joblib.load(MODEL_PATH)
    original_prediction = int(model.predict(normalize(original))[0])
    if original_prediction != 0:
        raise SystemExit(f"Baseline mismatch: expected RBF SVM prediction 0, got {original_prediction}.")

    # This specific region and +40 adjustment were chosen before evaluation.
    modified, difference = strengthen_stroke(original)
    modified_prediction = int(model.predict(normalize(modified))[0])
    scores = np.stack([
        model.decision_function(normalize(original))[0],
        model.decision_function(normalize(modified))[0],
    ])
    print("Verified actual digit: 5; original RBF SVM prediction: 0.")
    print("Edit: +40 (capped at 255), rows 5–8, columns 12–14, zero-based.")
    print("All pixels outside the approved region are EXACTLY unchanged.")
    print(f"Changed pixels: {np.count_nonzero(difference)} of 784")
    for row, column in np.argwhere(difference != 0):
        print(f"  ({row}, {column}): {original[row, column]} → {modified[row, column]}")
    print(f"\nOriginal prediction: {original_prediction}; modified prediction: {modified_prediction}")
    print("Decision scores — NOT probabilities (SVM scores are not percentages):")
    print(f"{'Digit':<8} {'Original':>12} {'Modified':>12} {'Change':>12}")
    for index, digit in enumerate(model.classes_):
        print(f"{digit:<8} {scores[0,index]:>12.6f} {scores[1,index]:>12.6f} "
              f"{scores[1,index] - scores[0,index]:>+12.6f}")
    print("\nThis is one fixed intervention on one drawing, not proof of a general cause.")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for axis, pixels, title in zip(
        axes[:2], [original, modified],
        [f"Original | prediction: {original_prediction}",
         f"Modified | prediction: {modified_prediction}"],
    ):
        axis.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(title)
        # Outlines are plot annotations only; they do not alter image pixels.
        axis.add_patch(Rectangle((11.5, 4.5), 3, 4, fill=False, edgecolor="cyan", linewidth=1))
        axis.axis("off")
    shown = axes[2].imshow(np.abs(difference), cmap="inferno", vmin=0, vmax=40,
                           interpolation="nearest")
    axes[2].set_title("Absolute pixel difference\n0–40 intensity levels")
    axes[2].axis("off")
    figure.colorbar(shown, ax=axes[2], fraction=0.046, pad=0.04)
    figure.suptitle("Actual digit: 5 | Same saved RBF SVM | Fixed local edit")
    figure.tight_layout()
    print("View the three-panel comparison window. Close it to exit.", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
