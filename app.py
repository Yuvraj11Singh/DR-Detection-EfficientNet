"""
app.py — Flask backend for RetinaAI DR Detection
Run: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from predict import load_model, predict_image

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # Allow cross-origin requests from the frontend

# Load model once at startup
print("[RetinaAI] Loading model...")
MODEL = load_model()
print("[RetinaAI] Model loaded. Server ready.")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Serve the frontend HTML."""
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart form-data with key 'image', OR JSON with base64 'image' field.
    Returns: JSON with prediction result.
    """
    start_time = time.time()

    # --- Handle file upload ---
    if "image" in request.files:
        file = request.files["image"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type."}), 400
        img_bytes = file.read()

    # --- Handle base64 JSON ---
    elif request.is_json and "image" in request.json:
        b64_data = request.json["image"]
        # Strip data URL prefix if present
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)

    else:
        return jsonify({"error": "No image provided. Send as multipart 'image' or base64 JSON."}), 400

    # --- Convert to PIL Image ---
    try:
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not open image: {str(e)}"}), 400

    # --- Run inference ---
    try:
        result = predict_image(MODEL, image)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    inference_time = round(time.time() - start_time, 3)
    result["inference_time_s"] = inference_time

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "EfficientNetB0"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"[RetinaAI] Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)