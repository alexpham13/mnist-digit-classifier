"""Evaluate both saved models on unique, locally collected handwriting."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image


PROJECT_DIRECTORY = Path(__file__).resolve().parent
EXAMPLES_DIRECTORY = PROJECT_DIRECTORY / "collected_examples"
MODEL_PATHS = {
    "SGDClassifier": PROJECT_DIRECTORY / "models" / "mnist_sgd.joblib",
    "RBF SVM": PROJECT_DIRECTORY / "models" / "mnist_svm.joblib",
}


def load_unique_examples(directory):
    """Validate pairs and group identical decoded pixels, not file names."""
    if not directory.is_dir():
        raise ValueError(f"Examples folder does not exist: {directory}")

    groups = {}
    referenced_images = set()
    valid_pairs = 0
    for metadata_path in sorted(directory.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
            actual = metadata.get("actual_digit")
            if type(actual) is not int or not 0 <= actual <= 9:
                raise ValueError("actual_digit must be an integer from 0 to 9")
            image_name = metadata.get("image_file")
            # Saved examples are local PNGs beside their metadata files.
            if (not isinstance(image_name, str)
                    or Path(image_name).name != image_name
                    or Path(image_name).suffix.lower() != ".png"):
                raise ValueError("image_file must name a PNG in this folder")
            image_path = directory / image_name
            referenced_images.add(image_path)
            with Image.open(image_path) as image:
                if image.format != "PNG" or image.mode != "L" or image.size != (28, 28):
                    raise ValueError("expected a 28x28 grayscale PNG saved by the app")
                pixels = np.array(image, dtype=np.uint8)
            if not pixels.any():
                raise ValueError("image is blank")
        except (OSError, ValueError) as error:
            print(f"WARNING: Skipping {metadata_path.name}: {error}")
            continue

        valid_pairs += 1
        fingerprint = hashlib.sha256(pixels.tobytes()).hexdigest()
        if fingerprint not in groups:
            groups[fingerprint] = {
                "pixels": pixels, "labels": set(), "files": [],
            }
        groups[fingerprint]["labels"].add(actual)
        groups[fingerprint]["files"].append(image_name)

    for image_path in sorted(directory.glob("*.png")):
        if image_path not in referenced_images:
            print(f"WARNING: PNG has no usable metadata reference: {image_path.name}")

    examples = []
    conflicts = 0
    for group in groups.values():
        if len(group["labels"]) != 1:
            # Never silently choose a ground-truth label when duplicates disagree.
            conflicts += 1
            print("WARNING: Excluding conflicting-label drawing: "
                  f"labels={sorted(group['labels'])}, files={group['files']}")
            continue
        examples.append((group["files"][0], group["pixels"], next(iter(group["labels"]))))

    print(f"\nValid PNG/JSON pairs: {valid_pairs}")
    print(f"Duplicate records removed: {valid_pairs - len(groups)}")
    print(f"Conflicting-label drawings excluded: {conflicts}")
    print(f"Unique drawings evaluated: {len(examples)}")
    return examples


def main():
    try:
        examples = load_unique_examples(EXAMPLES_DIRECTORY)
        if not examples:
            raise ValueError("No valid, uniquely labeled drawings available to evaluate.")

        # The app saves images AFTER cropping, resizing, and centering. Do not
        # perform those operations again: doing so could change the model input.
        # Restore exactly the app's float32 / 255.0 normalization and flat rows.
        images = np.stack([example[1] for example in examples])
        features = (images.astype(np.float32) / 255.0).reshape(-1, 784)
        actual = np.array([example[2] for example in examples], dtype=np.int64)

        predictions = {}
        for name, model_path in MODEL_PATHS.items():
            # Load only our trusted project models. No fit() or saving occurs.
            model = joblib.load(model_path)
            if model.n_features_in_ != 784:
                raise ValueError(f"{name} does not accept 784 pixel values")
            predictions[name] = model.predict(features)
    except Exception as error:
        raise SystemExit(f"Evaluation could not finish: {error}") from error

    sgd = predictions["SGDClassifier"]
    svm = predictions["RBF SVM"]
    print("\nDrawing  Actual  SGD prediction  Correct  SVM prediction  Correct")
    print("-" * 65)
    for index, label in enumerate(actual):
        print(f"{index + 1:<8} {label:<7} {sgd[index]:<15} "
              f"{'Yes' if sgd[index] == label else 'No':<8} "
              f"{svm[index]:<15} {'Yes' if svm[index] == label else 'No'}")

    print("\nDrawing file key:")
    for index, (filename, _, _) in enumerate(examples, start=1):
        print(f"  {index}: {filename}")

    print("\nModel accuracy on the same unique drawings:")
    for name, predicted in predictions.items():
        correct = int(np.count_nonzero(predicted == actual))
        print(f"  {name}: {correct / len(actual):.2%} ({correct}/{len(actual)} correct)")

    print("\nError counts by actual digit:")
    print(f"{'Digit':<8} {'Examples':<10} {'SGD errors':<12} {'SVM errors':<12}")
    for digit in range(10):
        mask = actual == digit
        count = int(mask.sum())
        if count == 0:
            print(f"{digit:<8} {0:<10} {'N/A':<12} {'N/A':<12}")
        else:
            sgd_errors = int(np.count_nonzero(sgd[mask] != digit))
            svm_errors = int(np.count_nonzero(svm[mask] != digit))
            print(f"{digit:<8} {count:<10} {sgd_errors:<12} {svm_errors:<12}")
    print("\nN/A means no examples for that digit. Results describe this collection,")
    print("not general handwriting accuracy, especially if errors were collected deliberately.")


if __name__ == "__main__":
    main()
