import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = ARTIFACTS_DIR / "garbage_classifier.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
LOW_CONFIDENCE_THRESHOLD = 0.55
DISPLAY_LABELS = {
    "plastic": "Plastic",
    "paper": "Paper",
    "glass": "Glass",
    "metal": "Metal",
    "organic_waste": "Organic Waste",
}

app = Flask(__name__)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


class Predictor:
    def __init__(self, model_path: Path, class_names_path: Path):
        # Cache the artifact paths and load them lazily so the app can recover
        # if training finishes after the Flask server has already started.
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = []
        self.last_loaded_mtime = 0.0

    def _artifact_mtime(self) -> float:
        if not self.model_path.exists() or not self.class_names_path.exists():
            return 0.0
        return max(self.model_path.stat().st_mtime, self.class_names_path.stat().st_mtime)

    def refresh(self) -> bool:
        # Reload the model only when artifacts are available and newer than the last load.
        current_mtime = self._artifact_mtime()
        if current_mtime == 0.0 or current_mtime <= self.last_loaded_mtime:
            return self.is_ready()

        # Inference-only load avoids requiring custom training losses/objects.
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        with self.class_names_path.open("r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        self.last_loaded_mtime = current_mtime
        return self.is_ready()

    def is_ready(self) -> bool:
        return self.model is not None and len(self.class_names) > 0

    def predict(self, image_path: Path) -> dict:
        # Keep inference input aligned with training model input.
        # The model graph already contains MobileNetV2 preprocessing.
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        probs = self.model.predict(img_array, verbose=0)[0]
        pred_index = int(np.argmax(probs))
        raw_label = self.class_names[pred_index]
        top_indices = np.argsort(probs)[::-1][:3]
        return {
            "label": DISPLAY_LABELS.get(raw_label, raw_label.replace("_", " ").title()),
            "raw_label": raw_label,
            "confidence": float(probs[pred_index]),
            "is_low_confidence": bool(probs[pred_index] < LOW_CONFIDENCE_THRESHOLD),
            "all_scores": {
                DISPLAY_LABELS.get(name, name.replace("_", " ").title()): float(score)
                for name, score in zip(self.class_names, probs)
            },
            "top_predictions": [
                {
                    "label": DISPLAY_LABELS.get(self.class_names[index], self.class_names[index].replace("_", " ").title()),
                    "confidence": float(probs[index]),
                }
                for index in top_indices
            ],
        }


predictor = Predictor(MODEL_PATH, CLASS_NAMES_PATH)


@app.route("/", methods=["GET"])
def index():
    predictor.refresh()
    return render_template("index.html", model_ready=predictor.is_ready())


@app.route("/predict", methods=["POST"])
def predict():
    if not predictor.refresh():
        return jsonify(
            {
                "success": False,
                "error": "Model not found. Please train the model first using model/train.py.",
            }
        ), 400

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No image file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Please select an image file."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {
                "success": False,
                "error": "Invalid file format. Please upload JPG, JPEG, PNG, BMP, or WEBP.",
            }
        ), 400

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid4().hex}_{safe_name}"
    image_path = UPLOAD_DIR / unique_name

    try:
        # Save to a temporary upload path, run inference, then delete the file.
        file.save(image_path)
        result = predictor.predict(image_path)
        return jsonify({"success": True, **result})
    except Exception as exc:
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500
    finally:
        if image_path.exists():
            os.remove(image_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
