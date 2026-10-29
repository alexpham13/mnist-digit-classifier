"""Replay one recorded drawing at two brush widths for a fair comparison."""

from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import joblib
from PIL import Image, ImageDraw, ImageTk

from digit_app import CANVAS_SIZE, make_input_preview, preprocess_drawing


MODEL_PATH = Path(__file__).resolve().parent / "models" / "mnist_svm.joblib"
BRUSH_WIDTHS = (20, 10)


def render_path(strokes, width):
    """Replay recorded coordinates without joining separate mouse strokes."""
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    pen = ImageDraw.Draw(image)
    for stroke in strokes:
        if not stroke:
            continue
        # Match digit_app.py: a mouse press leaves a dot, then each movement
        # draws a line with circular endpoints. Each stroke starts independently.
        segments = [(stroke[0], stroke[0])] + list(zip(stroke, stroke[1:]))
        for start, end in segments:
            pen.line([start, end], fill=255, width=width)
            radius = width / 2
            for x, y in (start, end):
                pen.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return image


class BrushWidthExperiment:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.strokes = []
        self.active_stroke = None
        self.drawing_photo = None
        self.preview_photos = []
        root.title("Brush Width Experiment — Same Path, Two Widths")
        root.geometry("1000x520")
        root.minsize(960, 500)
        root.resizable(True, True)

        tk.Label(root, text="Draw once, then compare the same path at 20 px and 10 px.",
                 font=("Arial", 16)).pack(pady=12)
        panels = tk.Frame(root)
        panels.pack(fill=tk.BOTH, expand=True, padx=12)
        drawing_panel = tk.Frame(panels)
        drawing_panel.grid(row=0, column=0, sticky="n", padx=10)
        tk.Label(drawing_panel, text="Draw here (20 px live view)").pack(pady=6)
        self.canvas = tk.Canvas(drawing_panel, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                background="black", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.continue_stroke)
        root.bind("<ButtonRelease-1>", self.end_stroke)

        self.preview_labels = []
        self.predictions = []
        for column, width in enumerate(BRUSH_WIDTHS, start=1):
            panels.columnconfigure(column, weight=1)
            panel = tk.Frame(panels)
            panel.grid(row=0, column=column, sticky="n", padx=10)
            tk.Label(panel, text=f"{width} px brush → processed 28×28").pack(pady=6)
            label = tk.Label(panel, text="Click Compare", background="black", foreground="white")
            label.pack()
            self.preview_labels.append(label)
            prediction = tk.StringVar(value="Prediction: —")
            tk.Label(panel, textvariable=prediction, font=("Arial", 22, "bold")).pack(pady=12)
            self.predictions.append(prediction)

        buttons = tk.Frame(root)
        buttons.pack(pady=10)
        tk.Button(buttons, text="Compare", command=self.compare).pack(side=tk.LEFT, padx=8)
        tk.Button(buttons, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=8)
        tk.Label(root, text="Same RBF SVM and app preprocessing for both. Nothing is saved.").pack(pady=(0, 12))

    def reset_results(self):
        for label, prediction in zip(self.preview_labels, self.predictions):
            label.config(image="", text="Click Compare")
            prediction.set("Prediction: —")
        self.preview_photos = []

    def refresh_drawing(self):
        # Render the live view from the recorded path too, so display and
        # comparison use the same geometry rather than independent drawing code.
        image = render_path(self.strokes, 20)
        self.drawing_photo = ImageTk.PhotoImage(image, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.drawing_photo, anchor=tk.NW)
        self.reset_results()

    def start_stroke(self, event):
        self.active_stroke = [(event.x, event.y)]
        self.strokes.append(self.active_stroke)
        self.refresh_drawing()

    def continue_stroke(self, event):
        if self.active_stroke is not None:
            self.active_stroke.append((event.x, event.y))
            self.refresh_drawing()

    def end_stroke(self, event):
        self.active_stroke = None

    def clear(self):
        self.strokes = []
        self.active_stroke = None
        self.refresh_drawing()

    def compare(self):
        # Freeze a single snapshot. Both renderings replay these exact points,
        # including the same stroke breaks; only brush width differs.
        snapshot = tuple(tuple(stroke) for stroke in self.strokes)
        try:
            features = [preprocess_drawing(render_path(snapshot, width))
                        for width in BRUSH_WIDTHS]
            if any(row is None for row in features):
                messagebox.showinfo("Draw a digit", "Draw a digit before clicking Compare.",
                                    parent=self.root)
                return
            predictions = [self.model.predict(row)[0] for row in features]
            photos = [ImageTk.PhotoImage(make_input_preview(row), master=self.root)
                      for row in features]
            self.preview_photos = photos  # Keep Tkinter image references alive.
            for label, result, photo, digit in zip(
                self.preview_labels, self.predictions, photos, predictions
            ):
                label.config(image=photo, text="")
                result.set(f"Prediction: {digit}")
        except Exception as error:
            messagebox.showerror("Comparison error", str(error), parent=self.root)


def main():
    try:
        model = joblib.load(MODEL_PATH)
        if model.n_features_in_ != 784 or getattr(model, "kernel", None) != "rbf":
            raise ValueError("Expected the saved RBF SVM with 784 input features.")
    except Exception as error:
        raise SystemExit(f"Could not load {MODEL_PATH}: {error}") from error
    root = tk.Tk()
    BrushWidthExperiment(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
