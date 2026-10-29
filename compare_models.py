"""Compare our saved linear SGD model with an RBF SVM on MNIST."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


PROJECT_DIRECTORY = Path(__file__).resolve().parent
SGD_PATH = PROJECT_DIRECTORY / "models" / "mnist_sgd.joblib"
SVM_PATH = PROJECT_DIRECTORY / "models" / "mnist_svm.joblib"


def evaluate_model(model, images, labels):
    """Measure accuracy and recall using the same test images for each model."""
    predictions = model.predict(images)
    accuracy = accuracy_score(labels, predictions)
    # Recall for digit 5, for example, is the fraction of actual 5s identified
    # correctly. Explicit labels keep the results ordered from 0 through 9.
    recalls = recall_score(
        labels, predictions, labels=np.arange(10), average=None, zero_division=0
    )
    return accuracy, recalls


def main():
    if not SGD_PATH.is_file():
        raise SystemExit(f"Cannot compare models: saved SGD model missing at {SGD_PATH}")

    # Read the existing model; never fit it again or overwrite its saved file.
    sgd = joblib.load(SGD_PATH)
    print("Loading MNIST from the project's dataset cache...", flush=True)
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
        data_home=str(PROJECT_DIRECTORY / "data"),
    )

    # Use exactly the same normalization and standard split as training.
    images = mnist.data.astype(np.float32) / 255.0
    labels = mnist.target.astype(np.int64)
    X_train, X_test = images[:60000], images[60000:]
    y_train, y_test = labels[:60000], labels[60000:]
    # The original downloaded values are no longer needed in memory.
    del mnist

    # Select only from training data. Stratification approximately preserves
    # the proportion of each digit; random_state makes the selection repeatable.
    subset_indices, _ = train_test_split(
        np.arange(len(y_train)),
        train_size=10000,
        stratify=y_train,
        random_state=42,
    )
    X_subset = X_train[subset_indices]
    y_subset = y_train[subset_indices]
    print(f"SVM training subset: {len(y_subset):,} images")
    print("Subset counts by digit:", np.bincount(y_subset, minlength=10))

    print("Evaluating saved SGD model...", flush=True)
    sgd_accuracy, sgd_recalls = evaluate_model(sgd, X_test, y_test)
    print(f"Saved SGD test accuracy: {sgd_accuracy:.2%}", flush=True)

    # Our SGD classifier uses linear boundaries. An RBF kernel instead measures
    # similarity to training examples, allowing nonlinear (curved) boundaries.
    # This flexibility costs more training time and memory. cache_size limits
    # the kernel cache to 256 MB, not the entire process's memory consumption.
    # gamma='scale' chooses the kernel's reach based on the training data.
    svm = SVC(kernel="rbf", gamma="scale", cache_size=256)
    print("Training RBF SVM on 10,000 images...", flush=True)
    start = perf_counter()
    svm.fit(X_subset, y_subset)
    print(f"SVM training time: {perf_counter() - start:.1f} seconds", flush=True)

    # Save separately so the drawing app continues using mnist_sgd.joblib.
    SVM_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(svm, SVM_PATH)
    print(f"Saved SVM to: {SVM_PATH}", flush=True)

    print("Evaluating SVM on the same 10,000 test images...", flush=True)
    svm_accuracy, svm_recalls = evaluate_model(svm, X_test, y_test)

    print("\nModel comparison (same 10,000-image test set)")
    print(f"{'Model':<22} {'Training images':>16} {'Test accuracy':>16}")
    print(f"{'Saved linear SGD':<22} {60000:>16,} {sgd_accuracy:>15.2%}")
    print(f"{'RBF SVM':<22} {len(y_subset):>16,} {svm_accuracy:>15.2%}")
    print("\nPer-digit recall: fraction of actual examples recognized correctly")
    print(f"{'Digit':<8} {'SGD recall':>14} {'SVM recall':>14}")
    for digit, (sgd_recall, svm_recall) in enumerate(zip(sgd_recalls, svm_recalls)):
        print(f"{digit:<8} {sgd_recall:>13.2%} {svm_recall:>13.2%}")
    print("\nTraining sizes differ: this compares practical baselines, not equal-data models.")
    print("The drawing application still uses the original SGD model.")


if __name__ == "__main__":
    main()
