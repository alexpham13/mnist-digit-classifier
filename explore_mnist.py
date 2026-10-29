"""Load MNIST, print its size, and display ten labeled handwritten digits."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def main():
    # Keep downloaded data inside this project instead of the home directory.
    data_directory = Path(__file__).resolve().parent / "data"
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
        data_home=str(data_directory),
    )

    # Each row contains 784 pixel values; the target is the correct digit.
    images = mnist.data
    labels = mnist.target
    image_height, image_width = 28, 28

    print(f"Total number of images: {images.shape[0]:,}")
    print(f"Dimensions of each image: {image_height} x {image_width} pixels")
    print(f"Pixel values per image: {images.shape[1]}")

    figure, axes = plt.subplots(2, 5, figsize=(10, 5))
    for index, axis in enumerate(axes.flat):
        # Turn a flat row of pixels back into a square image.
        digit_image = images[index].reshape(image_height, image_width)
        axis.imshow(digit_image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"Label: {labels[index]}")
        axis.axis("off")

    figure.suptitle("MNIST: First 10 Handwritten Digits")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
