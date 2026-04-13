"""
augment.py — Data augmentation pipeline for RetinaAI training
Provides augmentation layers and a dataset builder with augmentation applied.

Usage (standalone test):
    python augment.py --data_dir ./aptos_small --preview

Import in train.py:
    from augment import build_augmented_dataset, get_augmentation_layer
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ─── CLI (for preview mode) ───────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Augmentation pipeline preview")
parser.add_argument("--data_dir", type=str, default="./aptos_small")
parser.add_argument("--preview",  action="store_true", help="Show augmented sample grid")
parser.add_argument("--n_aug",    type=int, default=9, help="Number of augmented previews")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--batch",    type=int, default=32)
args, _ = parser.parse_known_args()


# ─── AUGMENTATION LAYER ───────────────────────────────────────────────────────

def get_augmentation_layer(
    rotation_factor: float = 0.15,
    zoom_factor: float = 0.1,
    flip_horizontal: bool = True,
    flip_vertical: bool = True,
    contrast_factor: float = 0.15,
    brightness_factor: float = 0.1,
) -> tf.keras.Sequential:
    """
    Returns a Keras Sequential model of augmentation layers.
    Applied only during training (layers are no-ops at inference time).

    Args:
        rotation_factor    : Max rotation as fraction of 2π (0.15 ≈ ±27°)
        zoom_factor        : Max zoom fraction
        flip_horizontal    : Random left-right flip
        flip_vertical      : Random up-down flip
        contrast_factor    : Random contrast adjustment range
        brightness_factor  : Random brightness adjustment range

    Returns:
        tf.keras.Sequential augmentation pipeline
    """
    layers = []

    if flip_horizontal:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    if flip_vertical:
        layers.append(tf.keras.layers.RandomFlip("vertical"))

    layers.extend([
        tf.keras.layers.RandomRotation(rotation_factor),
        tf.keras.layers.RandomZoom(zoom_factor),
        tf.keras.layers.RandomContrast(contrast_factor),
        tf.keras.layers.RandomBrightness(brightness_factor),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode="reflect",
        ),
    ])

    return tf.keras.Sequential(layers, name="augmentation")


# ─── AUGMENTED DATASET BUILDER ────────────────────────────────────────────────

def build_augmented_dataset(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    augment_train: bool = True,
    cache: bool = False,
) -> tuple:
    """
    Build training and validation tf.data.Dataset pipelines with optional augmentation.

    Args:
        data_dir      : Root directory with class subfolders
        img_size      : Image height/width
        batch_size    : Batch size
        val_split     : Fraction of data used for validation
        seed          : Random seed for reproducibility
        augment_train : Apply augmentation to training set
        cache         : Cache dataset in memory (fast but needs RAM)

    Returns:
        (train_ds, val_ds, class_names) tuple
    """
    load_kwargs = dict(
        validation_split=val_split,
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, subset="training", **load_kwargs
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, subset="validation", **load_kwargs
    )
    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    if augment_train:
        aug_layer = get_augmentation_layer()

        def augment_fn(image, label):
            image = aug_layer(image, training=True)
            return image, label

        train_ds = train_ds.map(augment_fn, num_parallel_calls=AUTOTUNE)

    if cache:
        train_ds = train_ds.cache()
        val_ds   = val_ds.cache()

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# ─── PREVIEW MODE ─────────────────────────────────────────────────────────────

def preview_augmentation(data_dir: str, n: int = 9, img_size: int = 224) -> None:
    """
    Load one image and show n augmented versions side by side.
    """
    # Grab one sample image
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, image_size=(img_size, img_size), batch_size=1, shuffle=True, seed=0
    )
    sample_image, sample_label = next(iter(ds))
    class_names = ds.class_names

    aug = get_augmentation_layer()

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(
        f"Augmentation Preview — Original class: {class_names[int(sample_label[0])]}",
        fontsize=14, fontweight="bold"
    )

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            aug_img = aug(sample_image, training=True)[0].numpy().astype("uint8")
            ax.imshow(aug_img)
            ax.set_title(f"Aug #{idx + 1}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("augmentation_preview.png", dpi=150)
    print("[augment] Saved: augmentation_preview.png")
    plt.show()


# ─── DATASET STATISTICS ───────────────────────────────────────────────────────

def print_dataset_stats(data_dir: str) -> dict:
    """
    Print class distribution and basic dataset statistics.
    """
    stats = {}
    total = 0
    print(f"\n[augment] Dataset statistics: {data_dir}")
    print(f"{'Class':<20} {'Images':>8}")
    print("─" * 30)

    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        n = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ])
        stats[cls] = n
        total += n
        print(f"  {cls:<18} {n:>8,}")

    print("─" * 30)
    print(f"  {'TOTAL':<18} {total:>8,}")

    if len(stats) == 2:
        vals = list(stats.values())
        ratio = max(vals) / (min(vals) + 1e-9)
        print(f"\n  Class imbalance ratio: {ratio:.2f}x")
        if ratio > 3:
            print("  ⚠ High imbalance detected — consider class_weight or oversampling.")

    print()
    return stats


if __name__ == "__main__":
    if os.path.exists(args.data_dir):
        print_dataset_stats(args.data_dir)
        if args.preview:
            preview_augmentation(args.data_dir, n=args.n_aug, img_size=args.img_size)
    else:
        print(f"[augment] Data directory not found: {args.data_dir}")
        print("[augment] Module loaded successfully. Import with: from augment import build_augmented_dataset")