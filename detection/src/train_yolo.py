import os
import shutil
import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


# -----------------------------------------------------------------------
# train_yolo.py  (updated for GTSRB dataset from Member A)
#
# The GTSRB archive from Member A has this structure:
#
#   archive_extracted/
#   ├── Train/
#   │   ├── 0/   ← class folders with .png images
#   │   ├── 1/
#   │   ...
#   │   └── 42/
#   ├── Test/
#   │   └── *.png
#   ├── Train.csv   ← Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
#   └── Test.csv    ← same columns
#
# Train.csv has bounding box (ROI) annotations for every training image,
# which is exactly what we need for YOLO object detection training.
#
# This script:
#   1. Reads Train.csv to get image paths + bounding boxes
#   2. Splits into train / val (85% / 15%)
#   3. Copies images as .jpg and writes YOLO .txt label files
#   4. Writes dataset.yaml
#   5. Fine-tunes YOLOv8n on the dataset
#   6. Saves best.pt → detection/models/best.pt
# -----------------------------------------------------------------------


# -----------------------------
# 1. Settings
# -----------------------------

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)   # repo root

# ── Dataset from Member A ──────────────────────────────────────────────
# Default: looks for archive_extracted/ next to the repo root
# Change ARCHIVE_DIR if you placed the data elsewhere.
ARCHIVE_DIR  = os.path.join(os.path.dirname(PROJECT_ROOT), "archive_extracted")

# Also support data placed directly in the repo's data/ folder
ALT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TRAIN_CSV    = os.path.join(ARCHIVE_DIR, "Train.csv")
TEST_CSV     = os.path.join(ARCHIVE_DIR, "Test.csv")
ARCHIVE_ROOT = ARCHIVE_DIR   # base for relative paths in the CSV

# ── Output YOLO dataset ───────────────────────────────────────────────
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR  = os.path.join(DATASET_DIR, "images")
LABELS_DIR  = os.path.join(DATASET_DIR, "labels")
YAML_PATH   = os.path.join(DATASET_DIR, "dataset.yaml")

# ── Model output ──────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Training hyperparameters ──────────────────────────────────────────
# CPU-optimised settings (trains in ~30-40 min on CPU):
#   - imgsz=320  : smaller images → much faster per-batch
#   - batch=32   : larger batch → better GPU/CPU utilisation
#   - epochs=10  : sufficient for a strong baseline
#   - fraction=0.3: use 30% of data → fast iteration
#
# If you have a GPU, change to: IMG_SIZE=640, BATCH_SIZE=16, EPOCHS=30, FRACTION=1.0
YOLO_BACKBONE = "yolov8n.pt"
EPOCHS        = 10
IMG_SIZE      = 320
BATCH_SIZE    = 32
VAL_SPLIT     = 0.15
FRACTION      = 0.3   # fraction of dataset to use (1.0 = full dataset, GPU recommended)

NUM_CLASSES   = 43

CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing (vehicles over 3.5t)",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5t prohibited",
    "No entry",
    "General caution",
    "Dangerous curve (left)",
    "Dangerous curve (right)",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows (right)",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all restrictions",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing (vehicles over 3.5t)",
]


# -----------------------------
# 2. Locate dataset
# -----------------------------

def find_dataset():
    """
    Try to find the GTSRB dataset. Checks:
      1. archive_extracted/ (Member A's zip extracted next to repo)
      2. data/ inside the repo
    Returns (train_csv_path, archive_base_path) or raises.
    """
    # Option 1: archive_extracted next to gfg folder
    if os.path.isfile(TRAIN_CSV):
        print(f"Dataset found at: {ARCHIVE_DIR}")
        return TRAIN_CSV, TEST_CSV, ARCHIVE_DIR

    # Option 2: data/ folder inside repo
    alt_train = os.path.join(ALT_DATA_DIR, "Train.csv")
    alt_test  = os.path.join(ALT_DATA_DIR, "Test.csv")
    if os.path.isfile(alt_train):
        print(f"Dataset found at: {ALT_DATA_DIR}")
        return alt_train, alt_test, ALT_DATA_DIR

    raise FileNotFoundError(
        "GTSRB dataset not found!\n"
        f"Expected at: {ARCHIVE_DIR}\n"
        f"Or at:       {ALT_DATA_DIR}\n\n"
        "Place Train.csv, Test.csv, Train/, Test/ there and re-run."
    )


# -----------------------------
# 3. bbox → YOLO format
# -----------------------------

def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Convert pixel bounding box to YOLO normalised (cx, cy, w, h).
    All values clamped to [0, 1].
    """
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    return cx, cy, w, h


# -----------------------------
# 4. Build YOLO dataset
# -----------------------------

def build_yolo_dataset(train_csv, archive_base):
    """
    Read GTSRB Train.csv, convert to YOLO format, split train/val,
    copy images and write label files, then write dataset.yaml.
    """

    # Skip if dataset already built
    if (os.path.isdir(os.path.join(IMAGES_DIR, "train"))
            and len(list(Path(os.path.join(IMAGES_DIR, "train")).glob("*.jpg"))) > 100):
        print(f"\n[INFO] YOLO dataset already exists at: {DATASET_DIR}")
        print("[INFO] Skipping conversion. Delete detection/dataset/ to rebuild.")
        return True

    print("\n--- Building YOLO Dataset from GTSRB Train.csv ---")

    # Create output directories
    for split in ("train", "val"):
        os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

    # Read Train.csv
    # Columns: Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
    df = pd.read_csv(train_csv)

    print(f"Total annotations in Train.csv: {len(df)}")
    print(f"Unique images: {df['Path'].nunique()}")
    print(f"Classes: {df['ClassId'].nunique()}")

    # Unique image paths
    unique_imgs = df["Path"].unique().tolist()

    # Train / val split on image level (stratify by class not possible here
    # since one image = one class, so we just random split)
    train_imgs, val_imgs = train_test_split(
        unique_imgs,
        test_size=VAL_SPLIT,
        random_state=42
    )

    train_set = set(train_imgs)
    val_set   = set(val_imgs)

    print(f"\nSplit: train={len(train_imgs)} | val={len(val_imgs)}")

    # Group annotations by image path
    grouped = df.groupby("Path")

    converted = 0
    skipped   = 0

    for img_rel_path, rows in grouped:

        # Determine split
        split = "train" if img_rel_path in train_set else "val"

        # Build absolute source path
        src_path = os.path.join(archive_base, img_rel_path)

        if not os.path.isfile(src_path):
            skipped += 1
            continue

        # Read image to get actual pixel dimensions
        img = cv2.imread(src_path)
        if img is None:
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]

        # Output paths
        stem    = Path(img_rel_path).stem
        # Make filename unique by including parent folder (class id)
        parent  = Path(img_rel_path).parent.name
        out_name = f"{parent}_{stem}"

        out_img = os.path.join(IMAGES_DIR, split, out_name + ".jpg")
        out_lbl = os.path.join(LABELS_DIR, split, out_name + ".txt")

        # Save image as JPEG
        cv2.imwrite(out_img, img)

        # Build YOLO label lines
        label_lines = []

        for _, row in rows.iterrows():

            x1 = int(row["Roi.X1"])
            y1 = int(row["Roi.Y1"])
            x2 = int(row["Roi.X2"])
            y2 = int(row["Roi.Y2"])
            class_id = int(row["ClassId"])

            if class_id >= NUM_CLASSES:
                continue

            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(out_lbl, "w") as f:
            f.write("\n".join(label_lines))

        converted += 1

        if converted % 5000 == 0:
            print(f"  Converted {converted} images ...")

    print(f"\nDone! Converted: {converted} | Skipped: {skipped}")

    # Write dataset.yaml
    dataset_cfg = {
        "path":  DATASET_DIR.replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    with open(YAML_PATH, "w") as f:
        yaml.dump(dataset_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"dataset.yaml written to: {YAML_PATH}")

    return True


# -----------------------------
# 5. Train YOLOv8
# -----------------------------

def train_yolo():
    """
    Fine-tune YOLOv8n on the GTSRB detection dataset.
    """

    if not os.path.exists(YAML_PATH):
        print("[ERROR] dataset.yaml not found. Run build_yolo_dataset() first.")
        return None

    print("\n--- Starting YOLOv8 Training ---")
    print(f"Backbone   : {YOLO_BACKBONE}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Image size : {IMG_SIZE}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"YAML       : {YAML_PATH}")

    model = YOLO(YOLO_BACKBONE)

    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        fraction=FRACTION,
        name="traffic_sign_detection",
        project=os.path.join(BASE_DIR, "runs"),
        exist_ok=True,
        patience=10,
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )

    # Copy best.pt to detection/models/
    best_src = os.path.join(
        BASE_DIR, "runs", "traffic_sign_detection", "weights", "best.pt"
    )

    if os.path.exists(best_src):
        best_dst = os.path.join(MODEL_DIR, "best.pt")
        shutil.copy2(best_src, best_dst)
        print(f"\n[SUCCESS] Best model saved to: {best_dst}")
    else:
        print("\n[WARNING] best.pt not found — check detection/runs/ manually.")

    return results


# -----------------------------
# 6. Entry point
# -----------------------------

if __name__ == "__main__":

    print("=" * 60)
    print(" YOLOv8 Traffic Sign Detection — Training Script")
    print(" Dataset: GTSRB (from Member A's archive.zip)")
    print("=" * 60)

    # Locate dataset
    train_csv_path, test_csv_path, archive_base = find_dataset()

    # Convert to YOLO format
    build_yolo_dataset(train_csv_path, archive_base)

    # Train YOLOv8
    train_yolo()

    print("\n[DONE] Training complete.")
    print(f"Model saved at: {os.path.join(MODEL_DIR, 'best.pt')}")
