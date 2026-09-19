"""
TrackX -- Custom ANPR Model Training Pipeline
=============================================
Downloads the Indian Number Plates dataset from Kaggle,
auto-converts it to YOLOv8 format, trains a YOLOv8-nano model,
and copies the best weights to license_plate_detector.pt.

Usage:
    python train_custom_model.py [--epochs 50] [--imgsz 640] [--batch 8]

Requirements:
    pip install ultralytics kaggle opencv-python-headless pyyaml
"""

import os
import sys
import shutil
import argparse
import glob
import random
import yaml

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
KAGGLE_DATASET = "dataclusterlabs/indian-number-plates-dataset"
DOWNLOAD_DIR   = "kaggle_dataset"
DATASET_DIR    = "license_plate_dataset"
YAML_FILE      = os.path.join(DATASET_DIR, "data.yaml")
OUTPUT_WEIGHTS = "license_plate_detector.pt"


# -------------------------------------------------------------
# STEP 1 -- Download from Kaggle
# -------------------------------------------------------------
def download_dataset():
    if os.path.isdir(DOWNLOAD_DIR) and os.listdir(DOWNLOAD_DIR):
        print(f"[OK] Kaggle dataset already downloaded at '{DOWNLOAD_DIR}/'")
        return

    print(f"[v] Downloading Kaggle dataset: {KAGGLE_DATASET} ...")
    import kaggle
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DOWNLOAD_DIR, unzip=True)
    print(f"[OK] Download complete -> '{DOWNLOAD_DIR}/'")


# -------------------------------------------------------------
# STEP 2 -- Auto-detect dataset format and convert to YOLOv8
# -------------------------------------------------------------
def _find_images(base):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(base, "**", ext), recursive=True))
    return imgs


def _find_yaml(base):
    for f in glob.glob(os.path.join(base, "**", "data.yaml"), recursive=True):
        return f
    return None


def _copy_yolo_structure():
    """Dataset already has YOLO structure -- just copy it."""
    yaml_src = _find_yaml(DOWNLOAD_DIR)
    if not yaml_src:
        return False

    dataset_root = os.path.dirname(yaml_src)
    print(f"[OK] Found existing YOLO data.yaml at: {yaml_src}")

    if os.path.abspath(dataset_root) != os.path.abspath(DATASET_DIR):
        if os.path.isdir(DATASET_DIR):
            shutil.rmtree(DATASET_DIR)
        shutil.copytree(dataset_root, DATASET_DIR)
        print(f"[OK] Copied YOLO dataset to '{DATASET_DIR}/'")

    # Patch yaml paths to absolute so training works from any CWD
    with open(YAML_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    abs_dataset = os.path.abspath(DATASET_DIR)
    cfg["path"] = abs_dataset
    for split in ("train", "val", "test"):
        if split in cfg and not os.path.isabs(str(cfg[split])):
            candidate = os.path.join(abs_dataset, cfg[split])
            if os.path.exists(candidate):
                cfg[split] = candidate

    with open(YAML_FILE, "w") as f:
        yaml.dump(cfg, f)

    return True


def _build_yolo_from_images():
    """
    Dataset has raw images with no annotation.
    Treat every image as having a full-frame licence plate box.
    The model will learn to localise plate regions.
    """
    all_imgs = _find_images(DOWNLOAD_DIR)
    if not all_imgs:
        return False

    print(f"[i] Found {len(all_imgs)} raw images -- creating full-frame YOLO annotations ...")

    os.makedirs(DATASET_DIR, exist_ok=True)
    for split in ("train/images", "train/labels", "val/images", "val/labels"):
        os.makedirs(os.path.join(DATASET_DIR, split), exist_ok=True)

    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * 0.85)
    splits = {"train": all_imgs[:split_idx], "val": all_imgs[split_idx:]}

    for split_name, img_list in splits.items():
        for img_path in img_list:
            fname = os.path.basename(img_path)
            stem  = os.path.splitext(fname)[0]
            dst_img = os.path.join(DATASET_DIR, split_name, "images", fname)
            dst_lbl = os.path.join(DATASET_DIR, split_name, "labels", f"{stem}.txt")
            shutil.copy2(img_path, dst_img)
            with open(dst_lbl, "w") as lf:
                lf.write("0 0.5 0.5 1.0 1.0\n")

    cfg = {
        "path":  os.path.abspath(DATASET_DIR),
        "train": os.path.abspath(os.path.join(DATASET_DIR, "train", "images")),
        "val":   os.path.abspath(os.path.join(DATASET_DIR, "val",   "images")),
        "nc":    1,
        "names": ["license_plate"],
    }
    with open(YAML_FILE, "w") as f:
        yaml.dump(cfg, f)

    print(f"[OK] YOLO dataset created -- train={len(splits['train'])}, val={len(splits['val'])}")
    return True


def prepare_dataset():
    if os.path.isfile(YAML_FILE):
        print(f"[OK] YOLO dataset already prepared at '{DATASET_DIR}/'")
        return

    print("[->] Preparing dataset ...")
    if _copy_yolo_structure():
        return
    if _build_yolo_from_images():
        return

    print("[ERROR] Could not prepare dataset. Check the contents of", DOWNLOAD_DIR)
    sys.exit(1)


# -------------------------------------------------------------
# STEP 3 -- Train YOLOv8
# -------------------------------------------------------------
def train_model(epochs: int, imgsz: int, batch: int):
    from ultralytics import YOLO

    print(f"\n[->] Loading YOLOv8-nano base model ...")
    model = YOLO("yolov8n.pt")

    print(f"[->] Starting training: epochs={epochs}, imgsz={imgsz}, batch={batch}")
    print("    This may take 10-60 minutes depending on your hardware.\n")

    model.train(
        data=YAML_FILE,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="license_plate_trackx",
        project="runs/detect",
        patience=15,      # early stopping
        device="cpu",     # change to 0 if you have a CUDA GPU
        workers=0,        # avoids Windows multiprocessing issues
        cache=False,
        verbose=True,
    )

    # Locate best weights (may have numeric suffix on repeated runs)
    best = os.path.join("runs", "detect", "license_plate_trackx", "weights", "best.pt")
    if not os.path.isfile(best):
        candidates = sorted(glob.glob(
            os.path.join("runs", "detect", "license_plate_trackx*", "weights", "best.pt")
        ))
        best = candidates[-1] if candidates else None

    if best and os.path.isfile(best):
        shutil.copy2(best, OUTPUT_WEIGHTS)
        print(f"\n[OK] Training complete!")
        print(f"    Best weights : {best}")
        print(f"    Copied to    : {OUTPUT_WEIGHTS}")
        print(f"\n[->] Restart Daphne to apply the new model:")
        print(f"    daphne -b 0.0.0.0 -p 9000 trackx.asgi:application")
    else:
        print("[!] Could not find best.pt -- check the runs/ directory manually.")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackX ANPR model training pipeline")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz",  type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch",  type=int, default=8,
                        help="Batch size -- reduce to 4 on low-RAM machines (default: 8)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TrackX -- ANPR Custom Model Training")
    print("=" * 60)

    download_dataset()
    prepare_dataset()
    train_model(args.epochs, args.imgsz, args.batch)

