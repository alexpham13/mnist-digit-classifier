"""Draw a digit and classify it with our existing MNIST model."""

from pathlib import Path
from datetime import datetime, timezone
import json
from uuid import uuid4
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageTk


CANVAS_SIZE = 280
BRUSH_WIDTH = 20
MODEL_PATH = Path(__file__).resolve().parent / "models" / "mnist_sgd.joblib"
MODEL_PATHS = {
    "SGDClassifier": MODEL_PATH,
    "RBF SVM": MODEL_PATH.with_name("mnist_svm.joblib"),
    "Augmented RBF SVM": MODEL_PATH.with_name("mnist_svm_augmented.joblib"),
}
# Friendly labels are separate from technical identifiers used in saved metadata.
MODEL_OPTIONS = {
    "Enhanced AI (Recommended)": "Augmented RBF SVM",
    "Classic AI": "RBF SVM",
    "Fast AI": "SGDClassifier",
}
MODEL_DESCRIPTIONS = {
    "Augmented RBF SVM": "Trained with varied digit positions and angles for more flexibility.",
    "RBF SVM": "Recognizes curved digit patterns using the original training examples.",
    "SGDClassifier": "A lightweight, fast model for basic digit recognition.",
}
EXAMPLES_DIRECTORY = Path(__file__).resolve().parent / "collected_examples"


def save_prediction_example(features, actual_digit, predicted_digit, model_name,
                            directory=EXAMPLES_DIRECTORY):
    """Save a lossless pixel image and matching metadata with a unique name."""
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    name = f"{timestamp:%Y%m%dT%H%M%S}_{uuid4().hex}"
    image_path = directory / f"{name}.png"
    metadata_path = directory / f"{name}.json"
    # Undo normalization only for PNG storage. Dividing these saved pixels by
    # 255 reconstructs the exact float32 input used by our models.
    pixels = np.rint(features.reshape(28, 28) * 255).astype(np.uint8)
    metadata = {
        "actual_digit": int(actual_digit),
        "predicted_digit": int(predicted_digit),
        "selected_model": model_name,
        "image_file": image_path.name,
        "saved_at_utc": timestamp.isoformat(),
        "normalization": "float32 pixel values divided by 255.0",
    }
    # Exclusive creation protects existing files even in the unlikely event
    # of a filename collision. Matching stems link the PNG and JSON files.
    with image_path.open("xb") as image_file:
        Image.fromarray(pixels).save(image_file, format="PNG")
    with metadata_path.open("x", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    return image_path


def make_input_preview(features):
    """Enlarge the model's actual pixels without smoothing or adding detail."""
    pixels = np.rint(features.reshape(28, 28) * 255).astype(np.uint8)
    return Image.fromarray(pixels).resize((280, 280), Image.Resampling.NEAREST)


def top_decision_scores(model, features):
    """Return the three highest-scoring digits for one processed drawing."""
    # Each digit gets a signed decision score. Higher means the model favors
    # that digit more. Scores can be negative and are NOT probabilities:
    # they are not percentages and do not need to add up to one.
    scores = model.decision_function(features)[0]
    ranked_indices = np.argsort(-scores, kind="stable")[:3]
    # Score columns follow classes_, so use it to find the correct digit labels.
    return [(model.classes_[index], float(scores[index])) for index in ranked_indices]


def preprocess_drawing(image):
    """Return one normalized row of 784 pixels, or None for a blank drawing."""
    grayscale = image.convert("L")
    bounds = grayscale.getbbox()
    if bounds is None:
        return None

    # Remove empty space, then fit inside 20 x 20 without stretching the digit.
    cropped = grayscale.crop(bounds)
    scale = 20 / max(cropped.size)
    resized = cropped.resize(
        (max(1, round(cropped.width * scale)),
         max(1, round(cropped.height * scale))),
        Image.Resampling.LANCZOS,
    )

    # MNIST uses a 28 x 28 frame with a margin around the handwriting.
    framed = Image.new("L", (28, 28), 0)
    framed.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))

    # Center the ink's center of mass: brighter pixels carry more weight.
    pixels = np.asarray(framed, dtype=np.float32)
    rows, columns = np.indices(pixels.shape)
    ink = pixels.sum()
    if ink == 0:
        return None
    center_x = (columns * pixels).sum() / ink
    center_y = (rows * pixels).sum() / ink
    offset = (round(13.5 - center_x), round(13.5 - center_y))
    centered = Image.new("L", (28, 28), 0)
    centered.paste(framed, offset)

    # Match training: floating-point values in [0, 1], one row per image.
    return (np.asarray(centered, dtype=np.float32) / 255.0).reshape(1, 784)


class DigitApp:
    def configure_styles(self):
        """Use the native ttk theme, with a small shared palette and font scale."""
        self.colors = {
            "background": "#f3f5f7", "surface": "#ffffff",
            "text": "#202938", "muted": "#586577", "accent": "#2459a6",
            "border": "#dde3eb", "highlight": "#eaf1fb",
            "diagnostics": "#f8fafc",
        }
        style = ttk.Style(self.root)
        # Keep the current platform theme so comboboxes retain native behavior.
        from tkinter import font as tkfont
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        self.ui_font = (family, 12)
        style.configure("App.TFrame", background=self.colors["background"])
        style.configure("Card.TFrame", background=self.colors["surface"])
        style.configure("App.Title.TLabel", background=self.colors["background"],
                        foreground=self.colors["text"], font=(family, 22, "bold"))
        style.configure("App.Subtitle.TLabel", background=self.colors["background"],
                        foreground=self.colors["muted"], font=self.ui_font)
        style.configure("Card.TLabel", background=self.colors["surface"],
                        foreground=self.colors["text"], font=self.ui_font)
        style.configure("Card.Heading.TLabel", background=self.colors["surface"],
                        foreground=self.colors["text"], font=(family, 13, "bold"))
        style.configure("Card.Muted.TLabel", background=self.colors["surface"],
                        foreground=self.colors["muted"], font=(family, 11))
        style.configure("Diagnostics.Muted.TLabel", background=self.colors["diagnostics"],
                        foreground=self.colors["muted"], font=(family, 11))
        style.configure("Save.Success.TLabel", background=self.colors["surface"],
                        foreground="#18733b", font=(family, 11, "bold"))
        style.configure("Save.Error.TLabel", background=self.colors["surface"],
                        foreground="#b42318", font=(family, 11, "bold"))
        style.configure("Prediction.TLabel", background=self.colors["surface"],
                        foreground=self.colors["accent"], font=(family, 32, "bold"))
        style.configure("Card.TCheckbutton", background=self.colors["surface"],
                        font=self.ui_font)
        style.configure("Action.TButton", padding=(16, 8), font=(family, 12, "bold"))
        style.configure("Secondary.TButton", padding=(12, 8), font=self.ui_font)

    def __init__(self, root):
        self.root = root
        root.title("MNIST Digit Classifier")
        root.resizable(True, True)
        root.geometry("1000x820")
        root.minsize(940, 780)
        self.previous_root_position = None
        root.bind("<Configure>", self.on_root_configure, add="+")
        self.configure_styles()
        root.configure(background=self.colors["background"])
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Allow vertical scrolling when the explanatory text or a small window
        # makes the full layout taller than the viewport. The drawing canvas
        # below remains a separate, fixed 280x280 surface.
        layout_canvas = tk.Canvas(root, background=self.colors["background"],
                                  highlightthickness=0, borderwidth=0)
        self.layout_canvas = layout_canvas
        layout_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=layout_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        layout_canvas.configure(yscrollcommand=scrollbar.set)
        content = ttk.Frame(layout_canvas, style="App.TFrame", padding=24)
        self.content = content
        content_window = layout_canvas.create_window(0, 0, window=content, anchor="nw")
        content.bind("<Configure>", lambda event: layout_canvas.configure(
            scrollregion=layout_canvas.bbox("all")), add="+")
        layout_canvas.bind("<Configure>", lambda event: layout_canvas.itemconfigure(
            content_window, width=event.width), add="+")
        content.columnconfigure(0, weight=1, uniform="panels")
        content.columnconfigure(1, weight=1, uniform="panels")
        content.rowconfigure(1, weight=1)
        header = ttk.Frame(content, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        ttk.Label(header, text="Handwritten Digit Classifier",
                  style="App.Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Draw a digit. Explore what the model sees.",
                  style="App.Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))

        drawing_panel = ttk.Frame(content, style="Card.TFrame", padding=20)
        drawing_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        drawing_panel.columnconfigure(0, weight=1)
        right_panel = ttk.Frame(content, style="Card.TFrame", padding=20)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        right_panel.columnconfigure(0, weight=1)
        self.models = {}
        self.model_errors = {}
        self.last_point = None
        self.preview_photo = None
        self.last_prediction = None

        ttk.Label(drawing_panel, text="Draw a digit", style="Card.Heading.TLabel").grid(
            row=0, column=0, sticky="w")
        model_controls = ttk.Frame(drawing_panel, style="Card.TFrame")
        model_controls.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        model_controls.columnconfigure(0, weight=1)
        ttk.Label(model_controls, text="Recognition model", style="Card.Muted.TLabel").grid(
            row=0, column=0, sticky="w")
        self.selected_model = tk.StringVar(value="Enhanced AI (Recommended)")
        self.model_selector = ttk.Combobox(
            model_controls, textvariable=self.selected_model,
            values=list(MODEL_OPTIONS), state="readonly", width=29,
        )
        self.model_selector.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.model_selector.bind("<<ComboboxSelected>>", self.on_model_changed)
        self.model_description = tk.StringVar()
        ttk.Label(model_controls, textvariable=self.model_description,
                  style="Card.Muted.TLabel", wraplength=340, justify=tk.LEFT).grid(
            row=2, column=0, sticky="w", pady=(8, 0))

        # This frame may expand, but the drawing surface must stay exactly
        # 280x280 so mouse coordinates still match the Pillow image pixel grid.
        canvas_frame = tk.Frame(drawing_panel, background=self.colors["border"], padx=1, pady=1)
        canvas_frame.grid(row=2, column=0, pady=(16, 0))
        self.canvas = tk.Canvas(
            canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
            background="black", highlightthickness=0, borderwidth=0,
        )
        self.canvas.pack()
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.continue_stroke)
        root.bind("<ButtonRelease-1>", self.end_stroke)
        ttk.Label(drawing_panel, text="Use your mouse to draw one digit, 0–9.",
                  style="Card.Muted.TLabel").grid(row=3, column=0, pady=(8, 0))

        buttons = ttk.Frame(drawing_panel, style="Card.TFrame")
        buttons.grid(row=4, column=0, sticky="ew", pady=(16, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        self.predict_button = ttk.Button(buttons, text="Predict", command=self.predict,
                                         style="Action.TButton")
        self.predict_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="Clear", command=self.clear,
                   style="Secondary.TButton").grid(row=0, column=1, sticky="ew", padx=(6, 0))

        example_controls = ttk.Frame(drawing_panel, style="Card.TFrame")
        example_controls.grid(row=5, column=0, sticky="ew", pady=(20, 0))
        example_controls.columnconfigure(1, weight=1)
        ttk.Separator(example_controls).grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(example_controls, text="Help Improve the AI", style="Card.Heading.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(16, 10))
        ttk.Label(example_controls, text=(
            "Everyone writes numbers differently! Save your handwriting to help build a collection "
            "of examples that could improve recognition of different handwriting styles in future versions."
        ), style="Card.Muted.TLabel", wraplength=340, justify=tk.LEFT).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ttk.Label(example_controls, text="Actual digit", style="Card.Muted.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 16), pady=(0, 6))
        self.actual_digit = tk.StringVar(value="")
        ttk.Combobox(example_controls, textvariable=self.actual_digit,
                     values=[str(digit) for digit in range(10)], state="readonly", width=7).grid(
            row=4, column=0, sticky="ew", padx=(0, 16), ipady=4)
        ttk.Button(example_controls, text="Save Handwriting Sample", command=self.save_example,
                   style="Secondary.TButton").grid(row=4, column=1, sticky="ew")
        self.save_feedback = tk.StringVar(value="")
        self.save_feedback_label = ttk.Label(
            example_controls, textvariable=self.save_feedback,
            style="Save.Success.TLabel", wraplength=340, justify=tk.LEFT,
        )
        self.save_feedback_label.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.save_feedback_label.grid_remove()
        self.actual_digit.trace_add("write", self.clear_save_feedback)
        ttk.Label(example_controls, text=(
            "Samples are saved locally. They are not uploaded or used to retrain the AI automatically."
        ), style="Card.Muted.TLabel", wraplength=340, justify=tk.LEFT).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        ttk.Label(right_panel, text="Recognition result", style="Card.Heading.TLabel").grid(
            row=0, column=0, sticky="w")
        self.result = tk.StringVar(value="Prediction: —")
        ttk.Label(right_panel, textvariable=self.result, style="Prediction.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 16))
        # Retain the label frame and label references expected by the existing
        # selection, ranking, and reset callbacks. Only their appearance changes.
        self.diagnostics = tk.LabelFrame(
            right_panel, text="Prediction diagnostics", background=self.colors["diagnostics"],
            foreground=self.colors["text"], font=(self.ui_font[0], 11, "bold"),
            relief=tk.FLAT, borderwidth=0, padx=12, pady=10,
            highlightthickness=1, highlightbackground=self.colors["border"],
            highlightcolor=self.colors["border"],
        )
        self.diagnostics.grid(row=2, column=0, sticky="ew")
        self.diagnostics.columnconfigure(0, weight=1)
        ttk.Label(self.diagnostics, text="Decision scores — NOT probabilities\nScore scales differ between models.",
                  style="Diagnostics.Muted.TLabel", wraplength=320, justify=tk.LEFT).grid(
            row=0, column=0, sticky="w", pady=(4, 12))
        self.score_labels = []
        for rank in range(3):
            label = tk.Label(
                self.diagnostics, text=f"{rank + 1}. —", anchor="w",
                font=(self.ui_font[0], 11, "bold" if rank == 0 else "normal"),
                background=self.colors["highlight"] if rank == 0 else self.colors["diagnostics"],
                foreground=self.colors["accent"] if rank == 0 else self.colors["text"],
                padx=10, pady=8,
            )
            label.grid(row=rank + 1, column=0, sticky="ew", pady=(0, 4))
            self.score_labels.append(label)

        self.show_input = tk.BooleanVar(value=False)
        ttk.Checkbutton(right_panel, text="Show Model Input", variable=self.show_input,
                        command=self.toggle_input_preview, style="Card.TCheckbutton").grid(
            row=3, column=0, sticky="w", pady=(16, 8))
        self.preview_frame = ttk.Frame(right_panel, style="Card.TFrame")
        self.preview_frame.grid(row=4, column=0, sticky="ew")
        self.preview_frame.columnconfigure(0, weight=1)
        ttk.Label(self.preview_frame, text="28×28 pixels · enlarged 10×",
                  style="Card.Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.preview_label = tk.Label(
            self.preview_frame, text="Click Predict to inspect the model input.",
            background=self.colors["surface"], foreground=self.colors["muted"],
            font=self.ui_font, borderwidth=0, padx=0, pady=0,
        )
        self.preview_label.grid(row=1, column=0)
        self.preview_frame.grid_remove()

        self.status = tk.StringVar(value="Use thick, clear strokes near the center.")
        ttk.Label(content, textvariable=self.status, style="App.Subtitle.TLabel",
                  wraplength=860, justify=tk.LEFT).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        self.load_models()
        self.on_model_changed()

    def on_root_configure(self, event):
        # Root bindings also receive child events; only track the main window.
        if event.widget is not self.root:
            return
        position = (self.root.winfo_rootx(), self.root.winfo_rooty())
        previous = self.previous_root_position
        self.previous_root_position = position
        if previous is None or position == previous:
            return
        # Dismiss the model popup without changing its selected value. This is
        # an internal Tk operation, so unsupported calls must remain harmless.
        try:
            self.root.tk.call("ttk::combobox::Unpost", str(self.model_selector))
        except tk.TclError:
            pass

    def toggle_input_preview(self):
        if self.show_input.get():
            self.preview_frame.grid()
        else:
            self.preview_frame.grid_remove()

    def reset_diagnostics(self):
        # Any edit, clear, or model change invalidates the old prediction.
        self.last_prediction = None
        for rank, label in enumerate(self.score_labels, start=1):
            label.config(text=f"{rank}. —")

    def update_input_preview(self, features):
        if features is None:
            self.preview_label.config(image="", text="No model input: draw a digit first.")
            self.preview_photo = None
            return
        # Use the same features object as predict(), not a second preprocessing pass.
        # Keep a reference so Tkinter does not discard the displayed image.
        self.preview_photo = ImageTk.PhotoImage(make_input_preview(features), master=self.root)
        self.preview_label.config(image=self.preview_photo, text="")

    def load_models(self):
        # Load each trusted project model once. A missing model does not prevent
        # the other one from working, and neither model is trained here.
        for name, path in MODEL_PATHS.items():
            try:
                model = joblib.load(path)
                if not callable(getattr(model, "decision_function", None)):
                    raise ValueError("Model does not support decision_function.")
                if model.n_features_in_ != 784:
                    raise ValueError("Model must accept 784 pixel values.")
                self.models[name] = model
            except FileNotFoundError:
                self.model_errors[name] = f"Model missing: {path.name}. Restore it and reopen the app."
            except Exception as error:
                self.model_errors[name] = f"Could not load {name}: {error}"

    def on_model_changed(self, event=None):
        display_name = self.selected_model.get()
        name = MODEL_OPTIONS[display_name]
        self.model_description.set(MODEL_DESCRIPTIONS[name])
        self.result.set("Prediction: —")
        self.reset_diagnostics()
        diagnostics_name = display_name.removesuffix(" (Recommended)")
        self.diagnostics.config(text=f"Prediction diagnostics — {diagnostics_name}")
        available = name in self.models
        self.predict_button.config(state=tk.NORMAL if available else tk.DISABLED)
        self.status.set(
            f"Using {display_name}. Draw a digit and click Predict." if available
            else self.model_errors.get(name, "Model unavailable.")
        )
        # Recompute from the current drawing so old-model scores never remain
        # visible after a switch. A blank canvas needs no popup on switching.
        if available and self.image.getbbox() is not None:
            self.predict()

    def clear_save_feedback(self, *args):
        self.save_feedback.set("")
        self.save_feedback_label.grid_remove()

    def show_save_feedback(self, message, success=False):
        self.save_feedback.set(message)
        self.save_feedback_label.configure(
            style="Save.Success.TLabel" if success else "Save.Error.TLabel"
        )
        self.save_feedback_label.grid()
        # After layout settles, reveal feedback automatically if a short window
        # places it below the viewport. The user need not scroll to find it.
        self.root.after_idle(self.reveal_save_feedback)

    def reveal_save_feedback(self):
        if not self.save_feedback.get():
            return
        self.root.update_idletasks()
        canvas = self.layout_canvas
        bottom = (self.save_feedback_label.winfo_rooty() - self.content.winfo_rooty()
                  + self.save_feedback_label.winfo_height() + 12)
        visible_bottom = canvas.canvasy(0) + canvas.winfo_height()
        if bottom > visible_bottom:
            bounds = canvas.bbox("all")
            if bounds and bounds[3] > 0:
                canvas.yview_moveto(max(0, (bottom - canvas.winfo_height()) / bounds[3]))

    def start_stroke(self, event):
        self.clear_save_feedback()
        self.last_point = (event.x, event.y)
        self.draw_segment(self.last_point, self.last_point)

    def continue_stroke(self, event):
        point = (event.x, event.y)
        if self.last_point is not None:
            self.draw_segment(self.last_point, point)
        self.last_point = point

    def end_stroke(self, event):
        self.last_point = None

    def draw_segment(self, start, end):
        self.canvas.create_line(*start, *end, fill="white", width=BRUSH_WIDTH,
                                capstyle=tk.ROUND)
        self.pen.line([start, end], fill=255, width=BRUSH_WIDTH)
        # Round endpoints also make a single mouse click leave a visible dot.
        radius = BRUSH_WIDTH / 2
        for x, y in (start, end):
            bounds = (x - radius, y - radius, x + radius, y + radius)
            self.canvas.create_oval(*bounds, fill="white", outline="white")
            self.pen.ellipse(bounds, fill=255)
        self.result.set("Prediction: —")
        self.reset_diagnostics()

    def clear(self):
        self.clear_save_feedback()
        self.canvas.delete("all")
        self.image.paste(0, (0, 0, CANVAS_SIZE, CANVAS_SIZE))
        self.last_point = None
        self.result.set("Prediction: —")
        self.reset_diagnostics()
        self.update_input_preview(None)

    def predict(self):
        self.reset_diagnostics()
        features = preprocess_drawing(self.image)
        # Refresh on every Predict, even while the preview is hidden.
        self.update_input_preview(features)
        if features is None:
            messagebox.showinfo("Draw a digit", "Draw a digit before clicking Predict.",
                                parent=self.root)
            return
        model_name = MODEL_OPTIONS[self.selected_model.get()]
        model = self.models.get(model_name)
        if model is None:
            return
        try:
            prediction = model.predict(features)[0]
            ranked_scores = top_decision_scores(model, features)
            # Store this exact input and its model together, rather than
            # preprocessing again when Save Example is clicked.
            self.last_prediction = (
                features.copy(), int(prediction), model_name
            )
            self.result.set(f"Prediction: {prediction}")
            for rank, (label, (digit, score)) in enumerate(
                zip(self.score_labels, ranked_scores), start=1
            ):
                label.config(text=f"{rank}. Digit {digit} | Decision score: {score:.4f}")
        except Exception as error:
            messagebox.showerror("Prediction error", str(error), parent=self.root)

    def save_example(self):
        self.clear_save_feedback()
        actual = self.actual_digit.get()
        if actual not in [str(digit) for digit in range(10)]:
            self.show_save_feedback("Select the actual digit (0–9) before saving.")
            return
        if self.last_prediction is None:
            self.show_save_feedback("Draw a digit and click Predict before saving.")
            return
        features, predicted, model_name = self.last_prediction
        try:
            image_path = save_prediction_example(features, actual, predicted, model_name)
        except OSError as error:
            self.show_save_feedback(f"Sample could not be saved.\n{error}")
            return
        self.status.set(f"Saved example: {image_path.name}")
        # The helper returns only after both PNG and JSON files are written.
        self.show_save_feedback(
            f"✓ Sample saved successfully!\nYour handwritten {actual} was saved locally.",
            success=True,
        )


def main():
    root = tk.Tk()
    DigitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
