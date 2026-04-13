"""
evaluate.py — Comprehensive model evaluation for RetinaAI
Generates full classification metrics, plots, and saves a JSON report.

Usage:
    python evaluate.py --model dr_model.h5 --data_dir ./aptos_small
"""

import argparse
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)

from predict import load_model, preprocess_image

# ─── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Evaluate DR Detection Model")
parser.add_argument("--model",     type=str, default="dr_model.h5",   help="Path to .h5 model")
parser.add_argument("--data_dir",  type=str, default="./aptos_small", help="Dataset directory")
parser.add_argument("--batch",     type=int, default=64,              help="Batch size")
parser.add_argument("--img_size",  type=int, default=224,             help="Image size")
parser.add_argument("--threshold", type=float, default=0.5,           help="Classification threshold")
parser.add_argument("--out_dir",   type=str, default="./eval_output", help="Directory to save outputs")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

print(f"\n[evaluate] Loading validation data from: {args.data_dir}")
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    args.data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(args.img_size, args.img_size),
    batch_size=args.batch,
)
class_names = val_data.class_names
val_data = val_data.prefetch(tf.data.AUTOTUNE)


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

print(f"[evaluate] Loading model: {args.model}")
model = load_model(args.model)


# ─── RUN PREDICTIONS ──────────────────────────────────────────────────────────

print("[evaluate] Running predictions...")
all_images, all_labels = [], []
for x_batch, y_batch in val_data:
    all_images.append(x_batch)
    all_labels.append(y_batch)

all_images = tf.concat(all_images, axis=0)
all_labels = tf.concat(all_labels, axis=0).numpy()

t0 = time.time()
pred_probs = model.predict(all_images, verbose=1).flatten()
inference_time = time.time() - t0
preds = (pred_probs >= args.threshold).astype(int)


# ─── CORE METRICS ─────────────────────────────────────────────────────────────

fpr, tpr, roc_thresholds = roc_curve(all_labels, pred_probs)
roc_auc = auc(fpr, tpr)

precision, recall, pr_thresholds = precision_recall_curve(all_labels, pred_probs)
avg_precision = average_precision_score(all_labels, pred_probs)

f1    = f1_score(all_labels, preds)
mcc   = matthews_corrcoef(all_labels, preds)
cm    = confusion_matrix(all_labels, preds)

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn + 1e-9)  # Recall / True Positive Rate
specificity = tn / (tn + fp + 1e-9)  # True Negative Rate
ppv         = tp / (tp + fp + 1e-9)  # Positive Predictive Value
npv         = tn / (tn + fn + 1e-9)  # Negative Predictive Value
accuracy    = (tp + tn) / len(all_labels)

report = classification_report(all_labels, preds, target_names=class_names, output_dict=True)

print(f"\n── EVALUATION RESULTS ──────────────────────────────")
print(f"  Accuracy      : {accuracy:.4f}")
print(f"  AUC-ROC       : {roc_auc:.4f}")
print(f"  Avg Precision : {avg_precision:.4f}")
print(f"  F1 Score      : {f1:.4f}")
print(f"  MCC           : {mcc:.4f}")
print(f"  Sensitivity   : {sensitivity:.4f}  (Recall / TPR)")
print(f"  Specificity   : {specificity:.4f}  (TNR)")
print(f"  PPV           : {ppv:.4f}  (Precision)")
print(f"  NPV           : {npv:.4f}")
print(f"  Inference time: {inference_time:.2f}s for {len(all_labels)} images")
print(f"────────────────────────────────────────────────────\n")
print(classification_report(all_labels, preds, target_names=class_names))


# ─── FIGURE 1: CONFUSION MATRIX ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Pastel1")
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
tick_marks = np.arange(len(class_names))
ax.set_xticks(tick_marks); ax.set_xticklabels(class_names)
ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
total = len(all_labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        v = cm[i, j]
        ax.text(j, i, f"{v}\n({v/total*100:.1f}%)", ha="center", va="center", fontsize=11)
ax.set_ylabel("Actual", fontsize=12)
ax.set_xlabel("Predicted", fontsize=12)
plt.tight_layout()
cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
print(f"[evaluate] Saved: {cm_path}")
plt.show()


# ─── FIGURE 2: ROC CURVE ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, linewidth=2, label=f"EfficientNetB0 (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.08)
ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(args.out_dir, "roc_curve.png")
plt.savefig(roc_path, dpi=150)
print(f"[evaluate] Saved: {roc_path}")
plt.show()


# ─── FIGURE 3: PRECISION-RECALL CURVE ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall, precision, linewidth=2, label=f"AP = {avg_precision:.3f}")
ax.fill_between(recall, precision, alpha=0.08)
ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pr_path = os.path.join(args.out_dir, "precision_recall.png")
plt.savefig(pr_path, dpi=150)
print(f"[evaluate] Saved: {pr_path}")
plt.show()


# ─── FIGURE 4: PREDICTION DISTRIBUTION ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(pred_probs[all_labels == 0], bins=40, alpha=0.65, label="No DR", color="#4fc3f7")
ax.hist(pred_probs[all_labels == 1], bins=40, alpha=0.65, label="DR",    color="#ef5350")
ax.axvline(args.threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {args.threshold}")
ax.set_title("Prediction Score Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Probability", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
dist_path = os.path.join(args.out_dir, "score_distribution.png")
plt.savefig(dist_path, dpi=150)
print(f"[evaluate] Saved: {dist_path}")
plt.show()


# ─── SAVE JSON REPORT ─────────────────────────────────────────────────────────

report_data = {
    "timestamp"        : datetime.now().isoformat(),
    "model_path"       : args.model,
    "dataset"          : args.data_dir,
    "threshold"        : args.threshold,
    "n_samples"        : int(len(all_labels)),
    "n_positive"       : int(all_labels.sum()),
    "n_negative"       : int((1 - all_labels).sum()),
    "accuracy"         : round(float(accuracy), 4),
    "auc_roc"          : round(float(roc_auc), 4),
    "avg_precision"    : round(float(avg_precision), 4),
    "f1_score"         : round(float(f1), 4),
    "mcc"              : round(float(mcc), 4),
    "sensitivity"      : round(float(sensitivity), 4),
    "specificity"      : round(float(specificity), 4),
    "ppv"              : round(float(ppv), 4),
    "npv"              : round(float(npv), 4),
    "tp"               : int(tp), "tn": int(tn),
    "fp"               : int(fp), "fn": int(fn),
    "inference_time_s" : round(inference_time, 3),
    "classification_report": report,
}

report_path = os.path.join(args.out_dir, "eval_report.json")
with open(report_path, "w") as f:
    json.dump(report_data, f, indent=2)
print(f"[evaluate] Saved: {report_path}")
print("\n[evaluate] Done.")