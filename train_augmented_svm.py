"""Train a separate augmented SVM and compare two fixed evaluation sets."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from evaluate_handwriting import load_unique_examples


PROJECT_DIRECTORY = Path(__file__).resolve().parent
ORIGINAL_PATH = PROJECT_DIRECTORY / "models" / "mnist_svm.joblib"
OUTPUT_PATH = PROJECT_DIRECTORY / "models" / "mnist_svm_augmented.joblib"
RANDOM_SEED = 42


def augment_images(images, seed=RANDOM_SEED):
    """Make one reproducible rotated-and-shifted copy per normalized image."""
    rng = np.random.default_rng(seed)
    augmented = np.empty_like(images)
    for index, row in enumerate(images):
        # Float-mode images preserve interpolated grayscale values without
        # rounding back to bytes. Inputs and outputs remain in the 0–1 range.
        image = Image.fromarray(row.reshape(28, 28))
        angle = float(rng.uniform(-10, 10))
        dx, dy = (int(value) for value in rng.integers(-2, 3, size=2))
        rotated = image.rotate(
            angle, resample=Image.Resampling.BILINEAR,
            expand=False, fillcolor=0,
        )
        # Paste clips pixels at the edges; it never wraps them around.
        shifted = Image.new("F", (28, 28), 0)
        shifted.paste(rotated, (dx, dy))
        augmented[index] = np.asarray(shifted, dtype=np.float32).reshape(784)
    return augmented


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(f"Refusing to overwrite an existing model: {OUTPUT_PATH}")
    original = joblib.load(ORIGINAL_PATH)
    # Match the original experiment, including SVC's remaining default settings.
    template = SVC(kernel="rbf", gamma="scale", cache_size=256)
    if original.get_params() != template.get_params():
        raise SystemExit("Saved SVM parameters differ from compare_models.py; stopping.")

    print("Loading MNIST from the project cache...", flush=True)
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff",
        data_home=str(PROJECT_DIRECTORY / "data"),
    )
    # Exactly the normalization and split used in compare_models.py.
    images = mnist.data.astype(np.float32) / 255.0
    labels = mnist.target.astype(np.int64)
    X_train, X_test = images[:60000], images[60000:]
    y_train, y_test = labels[:60000], labels[60000:]
    del mnist
    subset_indices, _ = train_test_split(
        np.arange(len(y_train)), train_size=10000,
        stratify=y_train, random_state=42,
    )
    X_subset = X_train[subset_indices]
    y_subset = y_train[subset_indices]
    print("Original subset counts:", np.bincount(y_subset, minlength=10))
    print("Creating one augmented copy per training image...", flush=True)
    augmented_images = augment_images(X_subset)
    training_images = np.concatenate([X_subset, augmented_images])
    training_labels = np.concatenate([y_subset, y_subset])
    # Mix originals and augmented copies reproducibly before fitting.
    order = np.random.default_rng(RANDOM_SEED).permutation(len(training_labels))
    training_images = training_images[order]
    training_labels = training_labels[order]
    del augmented_images

    augmented_model = clone(template)
    print(f"Training augmented SVM on {len(training_labels):,} images...", flush=True)
    start = perf_counter()
    augmented_model.fit(training_images, training_labels)
    print(f"Training completed in {perf_counter() - start:.1f} seconds.", flush=True)
    # Exclusive creation also protects the output if another run creates it
    # while training. The original model file is only ever read.
    with OUTPUT_PATH.open("xb") as model_file:
        joblib.dump(augmented_model, model_file)
    print(f"Saved new model: {OUTPUT_PATH}", flush=True)

    models = {"Original SVM": original, "Augmented SVM": augmented_model}
    print("\nUntouched MNIST test set (10,000 images):", flush=True)
    test_accuracies = {}
    for name, model in models.items():
        predicted = model.predict(X_test)
        accuracy = accuracy_score(y_test, predicted)
        test_accuracies[name] = accuracy
        print(f"  {name:<16} {accuracy:.2%} "
              f"({np.count_nonzero(predicted == y_test):,}/10,000 correct)", flush=True)
    difference = 100 * (test_accuracies["Augmented SVM"] - test_accuracies["Original SVM"])
    print(f"Augmentation changed MNIST accuracy by {difference:+.2f} percentage points.")

    print("\nSaved handwriting (evaluation only; never used for training or tuning):")
    examples_directory = PROJECT_DIRECTORY / "collected_examples"
    if not examples_directory.is_dir():
        print("No collected_examples folder; handwriting evaluation unavailable.")
        return
    examples = load_unique_examples(examples_directory)
    if not examples:
        print("No valid unique drawings; handwriting evaluation unavailable.")
        return
    # Saved PNGs are already cropped and centered. Restore their exact input
    # normalization without applying those geometric operations a second time.
    handwriting = np.stack([example[1] for example in examples])
    handwriting = (handwriting.astype(np.float32) / 255.0).reshape(-1, 784)
    actual = np.array([example[2] for example in examples], dtype=np.int64)
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(handwriting)
        correct = int(np.count_nonzero(predictions[name] == actual))
        print(f"  {name:<16} {correct / len(actual):.2%} ({correct}/{len(actual)} correct)")
    print("\nDrawing  Actual  Original SVM  Augmented SVM")
    for index, label in enumerate(actual):
        print(f"{index + 1:<8} {label:<7} {predictions['Original SVM'][index]:<13} "
              f"{predictions['Augmented SVM'][index]}")
    print("\nDrawing file key:")
    for index, (filename, _, _) in enumerate(examples, start=1):
        print(f"  {index}: {filename}")
    print("\nThe saved drawings are a small, previously inspected collection, possibly")
    print("biased toward errors. Their accuracy does not establish general performance.")
    print("Augmentation may help with position and tilt, but not every handwriting difference.")
    print("gamma='scale' uses the same rule but recalculates its value on expanded data.")


if __name__ == "__main__":
    main()
