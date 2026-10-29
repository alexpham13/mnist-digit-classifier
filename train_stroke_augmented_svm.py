"""Compare geometric augmentation with geometric-plus-stroke augmentation."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from evaluate_handwriting import load_unique_examples
from train_augmented_svm import RANDOM_SEED, augment_images


PROJECT_DIRECTORY = Path(__file__).resolve().parent
PREVIOUS_PATH = PROJECT_DIRECTORY / "models" / "mnist_svm_augmented.joblib"
OUTPUT_PATH = PROJECT_DIRECTORY / "models" / "mnist_svm_stroke_augmented.joblib"


def add_stroke_variations(images, seed=RANDOM_SEED):
    """Gently thin or thicken every geometric copy without adding more images."""
    # This generator is separate from the geometry generator so rotations and
    # shifts stay exactly the same as in the previous training experiment.
    rng = np.random.default_rng(seed)
    thicken = rng.integers(0, 2, size=len(images)).astype(bool)
    result = np.empty_like(images)
    for index, row in enumerate(images):
        pixels = row.reshape(28, 28)
        operation = grey_dilation if thicken[index] else grey_erosion
        transformed = operation(pixels, size=(3, 3), mode="constant", cval=0)
        # The preview's 75/25 blend, applied directly to normalized floats.
        # Avoid rounding to bytes so the geometric interpolation stays intact.
        result[index] = (0.75 * pixels + 0.25 * transformed).reshape(784)
    return result, thicken


def report_accuracy(models, features, actual, description):
    print(f"\n{description} ({len(actual):,} images):", flush=True)
    predictions = {}
    for name, model in models.items():
        predicted = model.predict(features)
        predictions[name] = predicted
        correct = int(np.count_nonzero(predicted == actual))
        print(f"  {name:<22} {correct / len(actual):.2%} "
              f"({correct:,}/{len(actual):,} correct)", flush=True)
    previous_accuracy = np.mean(predictions["Previous augmented SVM"] == actual)
    new_accuracy = np.mean(predictions["Stroke-augmented SVM"] == actual)
    print(f"Change: {100 * (new_accuracy - previous_accuracy):+.2f} percentage points.")
    return predictions


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(f"Refusing to overwrite existing model: {OUTPUT_PATH}")
    previous = joblib.load(PREVIOUS_PATH)
    template = SVC(kernel="rbf", gamma="scale", cache_size=256)
    if previous.get_params() != template.get_params():
        raise SystemExit("Previous model parameters differ from the expected setup.")

    print("Loading MNIST from the project cache...", flush=True)
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff",
        data_home=str(PROJECT_DIRECTORY / "data"),
    )
    images = mnist.data.astype(np.float32) / 255.0
    labels = mnist.target.astype(np.int64)
    X_train, X_test = images[:60000], images[60000:]
    y_train, y_test = labels[:60000], labels[60000:]
    del mnist

    # Exactly the subset selection from train_augmented_svm.py.
    subset_indices, _ = train_test_split(
        np.arange(len(y_train)), train_size=10000,
        stratify=y_train, random_state=42,
    )
    X_subset = X_train[subset_indices]
    y_subset = y_train[subset_indices]
    print("Original subset counts:", np.bincount(y_subset, minlength=10))
    geometric_copies = augment_images(X_subset, seed=RANDOM_SEED)
    stroke_copies, thicken = add_stroke_variations(geometric_copies)
    print(f"Gentle thinning: {np.count_nonzero(~thicken):,}; "
          f"gentle thickening: {np.count_nonzero(thicken):,}")
    training_images = np.concatenate([X_subset, stroke_copies])
    training_labels = np.concatenate([y_subset, y_subset])
    # Same shuffle as before: keep training order fixed across experiments.
    order = np.random.default_rng(RANDOM_SEED).permutation(len(training_labels))
    training_images = training_images[order]
    training_labels = training_labels[order]
    del geometric_copies, stroke_copies

    model = clone(template)
    print(f"Training on {len(training_labels):,} images...", flush=True)
    start = perf_counter()
    model.fit(training_images, training_labels)
    print(f"Training completed in {perf_counter() - start:.1f} seconds.", flush=True)
    # Exclusive creation prevents overwriting even if another run saves first.
    with OUTPUT_PATH.open("xb") as model_file:
        joblib.dump(model, model_file)
    print(f"Saved new model: {OUTPUT_PATH}", flush=True)

    models = {"Previous augmented SVM": previous, "Stroke-augmented SVM": model}
    report_accuracy(models, X_test, y_test, "Untouched MNIST test set")

    directory = PROJECT_DIRECTORY / "collected_examples"
    if not directory.is_dir():
        print("No collected_examples folder; handwriting evaluation unavailable.")
        return
    examples = load_unique_examples(directory)
    if not examples:
        print("No valid unique handwriting examples to evaluate.")
        return
    # Saved PNGs already have the app's geometric preprocessing. Restore the
    # original normalized input, without cropping or centering a second time.
    handwriting = np.stack([example[1] for example in examples])
    handwriting = (handwriting.astype(np.float32) / 255.0).reshape(-1, 784)
    actual = np.array([example[2] for example in examples], dtype=np.int64)
    predictions = report_accuracy(models, handwriting, actual, "Unique saved handwriting")
    print("\nDrawing  Actual  Previous SVM  Stroke SVM")
    for index, label in enumerate(actual):
        print(f"{index + 1:<8} {label:<7} "
              f"{predictions['Previous augmented SVM'][index]:<13} "
              f"{predictions['Stroke-augmented SVM'][index]}")
    print("\nDrawing file key:")
    for index, (filename, _, _) in enumerate(examples, start=1):
        print(f"  {index}: {filename}")
    print("\nOnly augmented copies received stroke changes; originals and test images")
    print("were unchanged. Saved handwriting was used only for evaluation.")
    print("The small, previously inspected handwriting collection cannot establish")
    print("general performance. Repeated test comparisons are exploratory; use a fresh")
    print("holdout for a final assessment. gamma='scale' recalculates its numeric value.")


if __name__ == "__main__":
    main()
