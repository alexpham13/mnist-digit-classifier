# MNIST Handwritten Digit Classifier

A Python 3.12 desktop application for drawing digits and exploring how machine
learning models recognize them. Built with Tkinter, scikit-learn, NumPy, and Pillow.

Draw a digit from **0–9**, choose a model, and inspect both its prediction and the
exact image it receives. The project also includes reproducible training scripts
and small experiments investigating handwriting style, brush width, and strokes.

## Application features

- A black **280×280** drawing canvas with thick white strokes.
- Three selectable models, with **Enhanced AI (Recommended)** selected at startup.
- A prominent prediction and the top three ranked **decision scores**.
- An optional **28×28 model-input preview**, enlarged without smoothing.
- **Predict**, **Clear**, and **Save Handwriting Sample** controls.
- A resizable two-column interface with scrolling for shorter windows.
- A macOS workaround that attempts to dismiss the model dropdown when the window
  moves. Its behavior can depend on the installed Tk version.

Decision scores are **not probabilities**, percentages, or calibrated confidence.
Higher scores rank a digit more favorably within a model; score scales differ
between models.

## Models

| Application option | Technical model | Required file |
|---|---|---|
| Enhanced AI (Recommended) | Rotation-and-shift augmented RBF SVM | `models/mnist_svm_augmented.joblib` |
| Classic AI | Original RBF SVM | `models/mnist_svm.joblib` |
| Fast AI | Linear SGDClassifier | `models/mnist_sgd.joblib` |

The experimental `models/mnist_svm_stroke_augmented.joblib` is separate and is not
used by the application dropdown.

**Pretrained models are not yet publicly downloadable.** Model binaries are
excluded from source control. A source-only checkout therefore cannot make
predictions until the required models are generated locally or supplied from a
trusted source. The application reports unavailable models rather than training
them automatically. Only load trusted joblib files; loading them can execute code.

## Setup and launch

Use **Python 3.12** with Tkinter support and a graphical desktop session. Tkinter
is provided by your Python/system installation, not by `pip install tkinter`.
If it is unavailable, install the matching Tk support for your Python distribution.

From the project directory on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import tkinter; print(tkinter.TkVersion)"
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell.

Once the model files listed above are present, launch:

```bash
python digit_app.py
```

Dependencies are pinned to versions used successfully in the existing development
environment. **A separate clean installation has not yet been verified.** Saved
scikit-learn models should be used with matching package versions; compatibility
across versions is not guaranteed.

The desktop app uses local model files and does not need to download MNIST.
Dataset scripts use `fetch_openml("mnist_784", version=1, as_frame=False,
parser="liac-arff")`, downloading data on first use and caching it in `data/`.
These scripts require internet access initially. If a download reports a TLS
certificate error, configure a trusted certificate bundle for your Python
installation; do not disable certificate verification.

### Generate the application models locally

The following commands **train models**; they are optional preparation, not part
of launching the application. Run them in order using the activated environment:

```bash
python train_classifier.py
python compare_models.py
python train_augmented_svm.py
```

1. `train_classifier.py` trains and saves `models/mnist_sgd.joblib`. Close its
   prediction plot window to finish the script.
2. `compare_models.py` loads that SGD model, trains the original SVM, and saves
   `models/mnist_svm.joblib`.
3. `train_augmented_svm.py` loads the original SVM and saves the enhanced model
   as `models/mnist_svm_augmented.joblib`.

The first two scripts can overwrite their respective outputs. The augmented
training scripts refuse to overwrite an existing output model. Back up any model
you need to preserve before deliberately rerunning training. Training time and
memory use vary by hardware; SVM training and evaluation can be substantially
more expensive than the linear classifier.

## Image preprocessing

The application maintains a grayscale Pillow image alongside the drawing canvas:

1. **Crop** to the nonblack pixel bounding box.
2. **Resize proportionally** to fit within 20×20 pixels using Lanczos interpolation.
3. Place the digit in a black **28×28** frame.
4. **Center the ink's weighted center** using an integer translation.
5. Convert to **float32**, divide pixel values by **255**, and flatten to a row
   of **784 values** for prediction.

This approximates MNIST formatting, but mouse strokes can still differ from the
training data. Blank drawings are handled without making a prediction.

## Saving handwriting samples

After predicting, select the **actual digit** and click **Save Handwriting Sample**.
Successful saves show a green confirmation beneath the button. Each sample creates
a uniquely named pair in the local, ignored `collected_examples/` directory:

- A lossless **28×28 grayscale PNG** containing the exact preprocessed input.
- A matching **JSON** file with `actual_digit`, `predicted_digit`, `selected_model`,
  `image_file`, `saved_at_utc`, and `normalization`.

The selected model is stored using its technical identifier. Editing the drawing
invalidates the previous prediction, so predict again before saving the new input.
Confirmation clears when starting another stroke, clicking Clear, or changing the
actual digit.

**Samples are saved locally. They are not uploaded or automatically used to
retrain any model.** Keep personal handwriting and metadata private unless you
explicitly choose to share them. No private examples are required to launch the app.

## Recorded evaluation results

All models below were evaluated on the same standard MNIST test split: the last
**10,000** images. Training uses only the first **60,000** images.

| Model | Training images | MNIST test accuracy |
|---|---|---:|
| SGDClassifier | 60,000 originals | 91.91% |
| Original RBF SVM | Stratified subset of 10,000 originals | 96.35% |
| Augmented RBF SVM | Same 10,000 originals + 10,000 geometric copies | 97.24% |
| Stroke-augmented RBF SVM | Same 10,000 originals + 10,000 geometric/stroke copies | 97.16% |

SVM experiments use `kernel="rbf"`, `gamma="scale"`, and a 256 MB kernel cache.
Subset selection and augmentation use fixed seeds. Geometric copies use rotations
up to ±10° and shifts up to ±2 pixels, with black padding and no wrapping.
Stroke augmentation blends 75% of each geometric copy with 25% grayscale erosion
or dilation. `gamma="scale"` recalculates its numeric value from each training set.

These are recorded development results, not a guarantee for arbitrary handwriting.
Mouse-drawn digits differ in thickness, shape, position, and style. In an early
evaluation of **four unique saved drawings, all labeled 5**, the original,
augmented, and stroke-augmented SVMs each recognized only **one of four (25%)**.
The local collection has since grown, so rerunning evaluation may produce different
results. It is small, selectively collected, and previously inspected; it is not
a representative independent handwriting benchmark.

Saved handwriting is used only for evaluation in these experiments, never as
training input. Repeated comparisons of the same test sets are exploratory; a
fresh holdout is needed for a final assessment after choosing improvements.

## Script guide

All scripts live in the project root. Running a script with `python script_name.py`
invokes its main function; importing the module does not start training or a GUI.

| Purpose | Scripts |
|---|---|
| Main desktop application | `digit_app.py` |
| Explore MNIST and display labeled images | `explore_mnist.py` |
| Train the linear baseline | `train_classifier.py` |
| Train the original SVM and compare accuracy/per-digit recall with SGD | `compare_models.py` |
| Train and evaluate geometric augmentation | `train_augmented_svm.py` |
| Train and evaluate geometric plus gentle stroke augmentation | `train_stroke_augmented_svm.py` |
| Evaluate both original models on deduplicated local handwriting | `evaluate_handwriting.py` |
| Compare a saved 5 with five MNIST test-set 5s | `compare_handwriting.py` |
| Preview rotations and shifts without training | `preview_augmentation.py` |
| Preview gentle erosion/dilation without training | `preview_stroke_thickness.py` |
| Replay one mouse path at 20 px and 10 px brush widths | `brush_width_experiment.py` |
| Fixed upper-stroke and bottom-tail pixel interventions | `stroke_hypothesis_test.py`, `bottom_tail_test.py` |

The handwriting evaluator deduplicates decoded image pixels and excludes groups
with conflicting actual-digit labels. Saved PNGs already have geometric
preprocessing applied, so evaluation restores normalization without cropping or
centering again.

The two fixed pixel-intervention scripts require one specific private saved PNG
and its metadata, which are not included in public source. They cannot run from
a source-only checkout. General handwriting comparisons need your own locally
saved samples. The brush-width experiment uses the original RBF SVM and keeps
drawings in memory.

## Local files excluded from publication

`.gitignore` excludes virtual environments, Python/tool caches, downloaded data,
generated models, private handwriting samples, common credential files, and
operating-system clutter. Recreate the environment locally rather than copying
another machine's `.venv/`. Review files explicitly before publishing; ignore
rules do not remove files that have already been committed.
