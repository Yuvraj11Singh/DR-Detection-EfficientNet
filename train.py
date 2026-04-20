"""
train.py — Standalone training script for DR Detection (EfficientNetB0)
Adapted from the original Google Colab notebook.

Usage:
    python train.py --data_dir ./aptos_small --epochs_head 5 --epochs_finetune 10
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# ─── CLI Arguments ────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train EfficientNetB0 for DR Detection")
parser.add_argument("--data_dir",       type=str,   default="./aptos_small",    help="Path to dataset directory")
parser.add_argument("--img_size",       type=int,   default=224,                help="Input image size")
parser.add_argument("--batch_size",     type=int,   default=64,                 help="Batch size")
parser.add_argument("--epochs_head",    type=int,   default=5,                  help="Epochs for head training")
parser.add_argument("--epochs_fine",    type=int,   default=10,                 help="Epochs for fine-tuning")
parser.add_argument("--unfreeze_layers",type=int,   default=30,                 help="Number of top layers to unfreeze")
parser.add_argument("--dropout",        type=float, default=0.4,                help="Dropout rate")
parser.add_argument("--output_model",   type=str,   default="dr_model.h5",      help="Where to save the trained model")
parser.add_argument("--val_split",      type=float, default=0.2,                help="Validation split ratio")
args = parser.parse_args()

IMG_SIZE    = args.img_size
BATCH       = args.batch_size
DATA_DIR    = args.data_dir
OUTPUT_PATH = args.output_model


# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────

print(f"\n[train] Loading data from: {DATA_DIR}")
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=args.val_split,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=args.val_split,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH
)

class_names = train_data.class_names
print(f"[train] Classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data   = val_data.prefetch(AUTOTUNE)


# ─── 2. BUILD MODEL ───────────────────────────────────────────────────────────

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

print("\n[train] Building EfficientNetB0 model...")
base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False   # Freeze for Phase 1

x   = GlobalAveragePooling2D()(base.output)
x   = Dropout(args.dropout)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(base.input, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
model.summary()


# ─── 3. PHASE 1 — TRAIN HEAD ──────────────────────────────────────────────────

print(f"\n[train] Phase 1: Training head for {args.epochs_head} epochs...")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=args.epochs_head
)


# ─── 4. PHASE 2 — FINE-TUNE TOP LAYERS ───────────────────────────────────────

print(f"\n[train] Phase 2: Unfreezing top {args.unfreeze_layers} layers for fine-tuning...")
for layer in base.layers[-args.unfreeze_layers:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=args.epochs_fine
)


# ─── 5. EVALUATION ────────────────────────────────────────────────────────────

print("\n[train] Evaluating on validation set...")
val_images, val_labels = [], []
for x_batch, y_batch in val_data:
    val_images.append(x_batch)
    val_labels.append(y_batch)

val_images = tf.concat(val_images, axis=0)
val_labels = tf.concat(val_labels, axis=0)

pred_probs = model.predict(val_images)
preds      = (pred_probs > 0.5).astype(int)

print("\n── CLASSIFICATION REPORT ──")
print(classification_report(val_labels, preds, target_names=class_names))


# ─── 6. CONFUSION MATRIX ──────────────────────────────────────────────────────

cm = confusion_matrix(val_labels, preds)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Pastel1")
ax.set_title("Confusion Matrix", fontsize=14)
tick_marks = np.arange(len(class_names))
ax.set_xticks(tick_marks); ax.set_xticklabels(class_names)
ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
total = np.sum(cm)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        ax.text(j, i, f"{val}\n({val/total*100:.2f}%)", ha="center", va="center", fontsize=11)
ax.set_ylabel("Actual", fontsize=12)
ax.set_xlabel("Predicted", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("[train] Saved: confusion_matrix.png")
plt.show()


# ─── 7. ROC CURVE ─────────────────────────────────────────────────────────────

fpr, tpr, _ = roc_curve(val_labels, pred_probs)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_title("ROC Curve", fontsize=14)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.legend(prop={"weight": "bold"})
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
print("[train] Saved: roc_curve.png")
plt.show()

print(f"\n[train] FINAL AUC: {roc_auc:.3f}")


# ─── 8. SAVE MODEL ────────────────────────────────────────────────────────────

model.save(OUTPUT_PATH)
print(f"[train] Model saved to: {OUTPUT_PATH}")

print("\n[train] Training complete!") 
print("You can now use the saved model for inference or further evaluation.")
print("Example usage:")
print(f"  python predict.py --model {OUTPUT_PATH} --image_path ./test_image.jpg")
