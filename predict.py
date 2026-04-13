"""
predict.py — Model loading and inference for RetinaAI
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import os

IMG_SIZE = 224
MODEL_PATH = os.environ.get("MODEL_PATH", "dr_model.h5")


def load_model(path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Load the trained EfficientNetB0 Keras model from disk.
    Falls back to building the architecture if no .h5 file is found (for dev/demo use).
    """
    if os.path.exists(path):
        print(f"[predict] Loading model from: {path}")
        model = keras_load_model(path)
    else:
        print(f"[predict] WARNING: {path} not found. Building untrained model for demo.")
        model = _build_model()

    return model


def _build_model() -> tf.keras.Model:
    """Recreate the EfficientNetB0 architecture (mirrors the Colab training script)."""
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model

    base = EfficientNetB0(
        weights=None,  # No weights for demo — use "imagenet" if you have internet
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(base.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and normalize a PIL image to match training preprocessing.
    EfficientNetB0 expects pixel values in [0, 255]; the layer handles rescaling.
    """
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32)          # shape (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)                 # shape (1, 224, 224, 3)
    return arr


def predict_image(model: tf.keras.Model, image: Image.Image) -> dict:
    """
    Run inference and return a structured result dict.

    Returns:
        {
            "dr_probability": float,       # Raw sigmoid output  [0, 1]
            "prediction": str,             # "DR Detected" | "No DR Detected"
            "label": str,                  # "Positive" | "Negative"
            "confidence": float,           # Confidence in the predicted class [0, 1]
            "confidence_pct": str,         # e.g. "87.4%"
        }
    """
    arr = preprocess_image(image)
    prob = float(model.predict(arr, verbose=0)[0][0])

    is_dr = prob > 0.5
    confidence = prob if is_dr else (1.0 - prob)

    return {
        "dr_probability": round(prob, 4),
        "prediction": "DR Detected" if is_dr else "No DR Detected",
        "label": "Positive" if is_dr else "Negative",
        "confidence": round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.1f}%",
        "model": "EfficientNetB0",
    }