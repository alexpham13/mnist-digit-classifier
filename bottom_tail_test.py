"""Test one approved three-pixel tail edit without changing saved files."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_DIRECTORY = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_DIRECTORY / "collected_examples" / (
    "20261028T165145_a49c1ed8fff5415fbf9aacb9f152f8b8.png"
)
MODEL_PATH = PROJECT_DIRECTORY / "models" / "mnist_svm.joblib"
APPROVED_PIXELS = {(12, 3): 90, (12, 4): 135, (12, 5): 12}


def normalize(pixels):
    # Saved pixels already have the app's crop, resize, and centering applied.
    # Only restore normalization; repeating geometry would change other pixels.
    return (pixels.astype(np.float32) / 255.0).reshape(1, 784)


def shorten_tail(original):
    """Verify all original values before making the edit in a separate array."""
    for (row, column), expected in APPROVED_PIXELS.items():
        actual = int(original[row, column])
        if actual != expected:
            raise ValueError(f"Pixel ({row}, {column}) mismatch: expected {expected}, got {actual}")
    modified = original.copy()
    mask = np.zeros(original.shape, dtype=bool)
    for row, column in APPROVED_PIXELS:
        mask[row, column] = True
        modified[row, column] = 0
    if not np.array_equal(modified[~mask], original[~mask]):
        raise ValueError("Pixels outside the approved three-pixel mask changed.")
    difference = np.abs(modified.astype(np.int16) - original.astype(np.int16))
    if not np.array_equal(difference != 0, mask):
        raise ValueError("The changed-pixel set differs from the approved mask.")
    return modified, difference


def main():
    try:
        metadata = json.loads(IMAGE_PATH.with_suffix(".json").read_text())
        if type(metadata.get("actual_digit")) is not int or metadata["actual_digit"] != 5:
            raise ValueError("Expected metadata actual_digit 5.")
        with Image.open(IMAGE_PATH) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (28, 28):
                raise ValueError("Expected a 28×28 grayscale PNG.")
            original = np.array(image)
        model = joblib.load(MODEL_PATH)
        original_prediction = int(model.predict(normalize(original))[0])
        if original_prediction != 0:
            raise ValueError(f"Baseline mismatch: expected prediction 0, got {original_prediction}.")
        modified, difference = shorten_tail(original)
        modified_prediction = int(model.predict(normalize(modified))[0])
        scores = np.stack([model.decision_function(normalize(pixels))[0]
                           for pixels in (original, modified)])
        index_0 = int(np.flatnonzero(model.classes_ == 0)[0])
        index_5 = int(np.flatnonzero(model.classes_ == 5)[0])
    except Exception as error:
        raise SystemExit(f"Experiment stopped: {error}") from error

    print("Verified actual digit 5, baseline prediction 0, and all three original pixel values.")
    for pixel, value in APPROVED_PIXELS.items():
        print(f"  Pixel {pixel}: {value} → 0")
    print("Exactly three pixels changed; all other 781 pixels are identical.")
    print("\nDecision scores — NOT probabilities. Gap = score(5) minus score(0).")
    print(f"{'Image':<12} {'Prediction':>12} {'Score 0':>14} {'Score 5':>14} {'Gap (5−0)':>14}")
    gaps = scores[:, index_5] - scores[:, index_0]
    for index, (name, prediction) in enumerate(
        [("Original", original_prediction), ("Modified", modified_prediction)]
    ):
        print(f"{name:<12} {prediction:>12} {scores[index,index_0]:>14.9f} "
              f"{scores[index,index_5]:>14.9f} {gaps[index]:>14.9f}")
    print(f"Gap change (modified minus original): {gaps[1] - gaps[0]:+.9f}")
    print("A positive gap change favors 5 relative to 0; a negative change favors 0.")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for axis, pixels, title in zip(
        axes[:2], (original, modified),
        (f"Original | prediction: {original_prediction}",
         f"Shortened tail | prediction: {modified_prediction}"),
    ):
        axis.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    shown = axes[2].imshow(difference, cmap="inferno", vmin=0, vmax=135,
                           interpolation="nearest")
    axes[2].set_title("Absolute pixel difference")
    axes[2].axis("off")
    figure.colorbar(shown, ax=axes[2], fraction=0.046, pad=0.04,
                    label="Intensity change")
    figure.suptitle("Actual digit: 5 | Same RBF SVM | Three-pixel tail test")
    figure.tight_layout()
    print("View the three-panel comparison window. No images or models were saved.", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
