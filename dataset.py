"""
dataset.py — Dataset preparation utilities for RetinaAI
Handles directory setup, class balancing, train/val splits, and image validation.

Usage:
    python dataset.py --data_dir ./aptos_small --balance --verify
"""

import argparse
import hashlib
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# ─── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Dataset preparation for RetinaAI")
parser.add_argument("--data_dir",       type=str,  default="./aptos_small", help="Dataset root directory")
parser.add_argument("--verify",         action="store_true",                help="Verify all images can be opened")
parser.add_argument("--balance",        action="store_true",                help="Balance classes by oversampling minority")
parser.add_argument("--remove_dupes",   action="store_true",                help="Remove duplicate images by hash")
parser.add_argument("--min_dim",        type=int,  default=100,             help="Minimum image dimension to keep")
parser.add_argument("--split",          action="store_true",                help="Create a pre-split directory structure")
parser.add_argument("--val_ratio",      type=float, default=0.2,            help="Validation split ratio")
parser.add_argument("--test_ratio",     type=float, default=0.1,            help="Test split ratio")
parser.add_argument("--seed",           type=int,  default=42,              help="Random seed")
args = parser.parse_args()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
random.seed(args.seed)
np.random.seed(args.seed)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_image_paths(directory: str) -> list:
    """Recursively collect all image paths under a directory."""
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def file_md5(path: str) -> str:
    """Return MD5 hash of a file (for duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_class_dirs(data_dir: str) -> list:
    """Return sorted list of class subdirectory paths."""
    return sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])


# ─── 1. DATASET OVERVIEW ──────────────────────────────────────────────────────

def print_overview(data_dir: str) -> dict:
    """Print a summary table of class sizes and return stats dict."""
    class_dirs = get_class_dirs(data_dir)
    stats = {}
    total = 0

    print(f"\n{'='*50}")
    print(f"  Dataset Overview: {data_dir}")
    print(f"{'='*50}")
    print(f"  {'Class':<22} {'Count':>8}  {'Share':>7}")
    print(f"  {'-'*40}")

    for cls_dir in class_dirs:
        cls  = os.path.basename(cls_dir)
        imgs = get_image_paths(cls_dir)
        n    = len(imgs)
        stats[cls] = {"count": n, "paths": imgs}
        total += n

    for cls, info in stats.items():
        n = info["count"]
        share = n / total * 100 if total > 0 else 0
        print(f"  {cls:<22} {n:>8,}  {share:>6.1f}%")

    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<22} {total:>8,}  {'100.0%':>7}")

    if len(stats) >= 2:
        counts = [v["count"] for v in stats.values()]
        ratio = max(counts) / (min(counts) + 1e-9)
        print(f"\n  Imbalance ratio : {ratio:.2f}x")
        print(f"  Num classes     : {len(stats)}")

    print(f"{'='*50}\n")
    return stats


# ─── 2. IMAGE VERIFICATION ────────────────────────────────────────────────────

def verify_images(data_dir: str, min_dim: int = 100) -> dict:
    """
    Open every image file, check it's valid and above min_dim resolution.
    Reports corrupt, too-small, and grayscale images.
    """
    all_paths = get_image_paths(data_dir)
    results   = {"ok": 0, "corrupt": [], "too_small": [], "grayscale": []}

    print(f"[dataset] Verifying {len(all_paths)} images...")
    for path in all_paths:
        try:
            img = Image.open(path)
            img.verify()  # Detects corruption
            img = Image.open(path)  # Re-open after verify
            w, h = img.size
            if w < min_dim or h < min_dim:
                results["too_small"].append(path)
            elif img.mode not in ("RGB", "RGBA"):
                results["grayscale"].append(path)
            else:
                results["ok"] += 1
        except (UnidentifiedImageError, Exception):
            results["corrupt"].append(path)

    print(f"\n── Verification Report ────────────────────────────")
    print(f"  OK         : {results['ok']}")
    print(f"  Corrupt    : {len(results['corrupt'])}")
    print(f"  Too small  : {len(results['too_small'])}  (< {min_dim}px)")
    print(f"  Grayscale  : {len(results['grayscale'])}")

    if results["corrupt"]:
        print("\n  Corrupt files:")
        for p in results["corrupt"][:10]:
            print(f"    {p}")

    print(f"──────────────────────────────────────────────────\n")
    return results


# ─── 3. DUPLICATE REMOVAL ─────────────────────────────────────────────────────

def remove_duplicates(data_dir: str, dry_run: bool = True) -> list:
    """
    Find and optionally delete duplicate images using MD5 hashing.

    Args:
        data_dir : Dataset root
        dry_run  : If True, only report duplicates without deleting

    Returns:
        List of duplicate file paths
    """
    all_paths = get_image_paths(data_dir)
    hash_map  = {}
    dupes     = []

    print(f"[dataset] Scanning {len(all_paths)} images for duplicates...")
    for path in all_paths:
        h = file_md5(path)
        if h in hash_map:
            dupes.append(path)
        else:
            hash_map[h] = path

    print(f"[dataset] Found {len(dupes)} duplicate(s).")

    if dupes and not dry_run:
        for path in dupes:
            os.remove(path)
            print(f"  Deleted: {path}")
        print(f"[dataset] Removed {len(dupes)} duplicate files.")
    elif dupes:
        print("[dataset] Dry run — pass dry_run=False to delete.")

    return dupes


# ─── 4. CLASS BALANCING (OVERSAMPLING) ───────────────────────────────────────

def balance_classes(data_dir: str, strategy: str = "oversample") -> None:
    """
    Balance class sizes by copying minority class images (oversampling)
    or removing majority class images (undersampling).

    Args:
        data_dir  : Dataset root (modified in-place)
        strategy  : "oversample" (default) or "undersample"
    """
    class_dirs = get_class_dirs(data_dir)
    class_info = {}
    for cls_dir in class_dirs:
        cls = os.path.basename(cls_dir)
        class_info[cls] = get_image_paths(cls_dir)

    sizes  = {cls: len(paths) for cls, paths in class_info.items()}
    target = max(sizes.values()) if strategy == "oversample" else min(sizes.values())

    print(f"[dataset] Balancing classes — strategy: {strategy}, target: {target:,} per class")

    for cls, paths in class_info.items():
        current = len(paths)
        if current == target:
            print(f"  {cls}: already at {current:,}. No change.")
            continue

        cls_dir = os.path.join(data_dir, cls)

        if strategy == "oversample" and current < target:
            diff   = target - current
            extras = random.choices(paths, k=diff)
            for i, src in enumerate(extras):
                ext  = Path(src).suffix
                dest = os.path.join(cls_dir, f"aug_copy_{i:06d}{ext}")
                shutil.copy2(src, dest)
            print(f"  {cls}: {current:,} → {target:,}  (+{diff:,} copies)")

        elif strategy == "undersample" and current > target:
            remove = random.sample(paths, current - target)
            for path in remove:
                os.remove(path)
            print(f"  {cls}: {current:,} → {target:,}  (-{current-target:,} removed)")

    print("[dataset] Balancing complete.\n")


# ─── 5. TRAIN / VAL / TEST SPLIT ─────────────────────────────────────────────

def create_split(
    data_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Reorganize a flat class-folder dataset into train/val/test splits.

    Input structure:
        data_dir/ClassA/img1.jpg ...
        data_dir/ClassB/img1.jpg ...

    Output structure:
        output_dir/train/ClassA/
        output_dir/val/ClassA/
        output_dir/test/ClassA/
    """
    random.seed(seed)
    class_dirs = get_class_dirs(data_dir)
    splits     = ["train", "val", "test"]

    # Create output directories
    for split in splits:
        for cls_dir in class_dirs:
            cls = os.path.basename(cls_dir)
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    print(f"[dataset] Creating splits → {output_dir}")

    for cls_dir in class_dirs:
        cls   = os.path.basename(cls_dir)
        paths = get_image_paths(cls_dir)
        random.shuffle(paths)

        n       = len(paths)
        n_test  = int(n * test_ratio)
        n_val   = int(n * val_ratio)
        n_train = n - n_test - n_val

        split_map = {
            "test" : paths[:n_test],
            "val"  : paths[n_test: n_test + n_val],
            "train": paths[n_test + n_val:],
        }

        for split, split_paths in split_map.items():
            dest_dir = os.path.join(output_dir, split, cls)
            for src in split_paths:
                shutil.copy2(src, dest_dir)
            print(f"  {split}/{cls}: {len(split_paths):,} images")

    print("[dataset] Split complete.\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(args.data_dir):
        print(f"[dataset] Directory not found: {args.data_dir}")
    else:
        stats = print_overview(args.data_dir)

        if args.verify:
            verify_images(args.data_dir, min_dim=args.min_dim)

        if args.remove_dupes:
            remove_duplicates(args.data_dir, dry_run=False)

        if args.balance:
            balance_classes(args.data_dir, strategy="oversample")
            print_overview(args.data_dir)

        if args.split:
            split_dir = args.data_dir.rstrip("/") + "_split"
            create_split(
                args.data_dir, split_dir,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )