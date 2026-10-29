"""Train a first MNIST classifier, evaluate it, and save it for later use."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def main():
    project_directory = Path(__file__).resolve().parent
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
        data_home=str(project_directory / "data"),
    )

    # Floating-point numbers preserve fractions when we divide by 255.
    # Black pixels become 0.0 and the brightest pixels become 1.0.
    images = mnist.data.astype(np.float32) / 255.0
    labels = mnist.target.astype(np.int64)

    # Preserve MNIST's standard order: 60,000 training and 10,000 test images.
    # The test images stay out of training so they provide a fair evaluation.
    X_train, X_test = images[:60000], images[60000:]
    y_train, y_test = labels[:60000], labels[60000:]

    classifier = SGDClassifier(random_state=42)
    print(f"Training on {len(X_train):,} images...", flush=True)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy on {len(X_test):,} images: {accuracy:.2%}")

    # Save before opening the plot window. Future inputs must also be divided
    # by 255 and arranged as rows of 784 floating-point pixel values.
    model_directory = project_directory / "models"
    model_directory.mkdir(exist_ok=True)
    model_path = model_directory / "mnist_sgd.joblib"
    joblib.dump(classifier, model_path)
    print(f"Saved trained model to: {model_path}")

    figure, axes = plt.subplots(2, 5, figsize=(12, 6))
    for index, axis in enumerate(axes.flat):
        axis.imshow(X_test[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"Predicted: {predictions[index]}\nActual: {y_test[index]}"
        )
        axis.axis("off")

    figure.suptitle("MNIST: Predictions for the First 10 Test Images")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
