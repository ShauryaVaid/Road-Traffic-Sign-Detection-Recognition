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
# This script converts the GTSDB (German Traffic Sign Detection Benchmark)
# dataset to YOLO format and trains a YOLOv8 model on it.
#
# GTSDB dataset structure expected:
#
#   data/
#   └── gtsdb/
#       ├── Images/          ← all .ppm images (00000.ppm … 00599.ppm)
#       └── gt.txt           ← ground-truth CSV (no header):
#                               filename;x1;y1;x2;y2;class_id
#
# After conversion the script creates:
#
#   detection/
#   └── dataset/
#       ├── images/
#       │   ├── train/
#       │   └── val/
#       ├── labels/
#       │   ├── train/
#       │   └── val/
#       └── dataset.yaml
# -----------------------------------------------------------------------


# -----------------------------
# 1. Settings
# -----------------------------

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)   # repo root

# Raw GTSDB data (user must place it here)
RAW_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "gtsdb", "Images")
GT_FILE        = os.path.join(PROJECT_ROOT, "data", "gtsdb", "gt.txt")

# Output YOLO dataset
DATASET_DIR    = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR     = os.path.join(DATASET_DIR, "images")
LABELS_DIR     = os.path.join(DATASET_DIR, "labels")
YAML_PATH      = os.path.join(DATASET_DIR, "dataset.yaml")

# Trained model output
MODEL_DIR      = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# YOLOv8 pre-trained backbone to fine-tune
YOLO_BACKBONE  = "yolov8n.pt"   # nano (fast), change to yolov8s.pt for better accuracy

# Training hyperparameters
EPOCHS      = 50
IMG_SIZE    = 640
BATCH_SIZE  = 16
VAL_SPLIT   = 0.15

NUM_CLASSES = 43

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
# 2. Convert pixel bbox → YOLO
# -----------------------------

def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Convert pixel-coordinate bounding box to YOLO normalised format.

    YOLO format: cx cy w h (all values between 0 and 1)

    Parameters
    ----------
    x1, y1, x2, y2 : int
        Pixel coordinates of the bounding box (top-left, bottom-right).
    img_w, img_h : int
        Width and height of the image in pixels.

    Returns
    -------
    tuple[float, float, float, float]
        (cx, cy, w, h) normalised to [0, 1].
    """
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    return cx, cy, w, h


# -----------------------------
# 3. Build YOLO dataset
# -----------------------------

def build_yolo_dataset():
    """
    Read GTSDB gt.txt, convert all annotations to YOLO format,
    split into train/val sets, copy images, and write label .txt files.
    Also generates dataset.yaml.
    """

    # -- Validate raw data paths --
    if not os.path.isdir(RAW_IMAGES_DIR):
        print(
            f"\n[WARNING] GTSDB images folder not found at:\n  {RAW_IMAGES_DIR}\n"
            "Skipping dataset conversion. If you already have the YOLO dataset\n"
            "in detection/dataset/, training will use that directly."
        )
        return False

    if not os.path.isfile(GT_FILE):
        print(
            f"\n[WARNING] GTSDB gt.txt not found at:\n  {GT_FILE}\n"
            "Skipping dataset conversion."
        )
        return False

    print("\n--- Building YOLO Dataset ---")

    # -- Read ground truth file --
    # GTSDB gt.txt format: Filename;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
    gt = pd.read_csv(
        GT_FILE,
        sep=";",
        header=None,
        names=["filename", "x1", "y1", "x2", "y2", "class_id"]
    )

    print(f"Total annotations: {len(gt)}")
    print(f"Unique images with annotations: {gt['filename'].nunique()}")

    # -- Gather all image filenames (annotated + unannotated) --
    all_images = [
        f for f in os.listdir(RAW_IMAGES_DIR)
        if f.lower().endswith((".ppm", ".jpg", ".jpeg", ".png"))
    ]

    print(f"Total images in dataset: {len(all_images)}")

    # -- Create output directory structure --
    for split in ("train", "val"):
        os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

    # -- Train / val split on image list --
    train_imgs, val_imgs = train_test_split(
        all_images,
        test_size=VAL_SPLIT,
        random_state=42
    )

    print(f"Train images: {len(train_imgs)} | Val images: {len(val_imgs)}")

    # -- Group annotations by filename --
    gt_grouped = gt.groupby("filename")

    # -- Process each image --
    converted = 0

    for img_filename in all_images:

        split = "train" if img_filename in train_imgs else "val"

        src_img_path = os.path.join(RAW_IMAGES_DIR, img_filename)

        # Read image to get dimensions
        img = cv2.imread(src_img_path)

        if img is None:
            print(f"  [SKIP] Cannot read: {img_filename}")
            continue

        img_h, img_w = img.shape[:2]

        # Save as .jpg in the appropriate split folder
        stem    = Path(img_filename).stem
        out_img = os.path.join(IMAGES_DIR, split, stem + ".jpg")
        out_lbl = os.path.join(LABELS_DIR, split, stem + ".txt")

        cv2.imwrite(out_img, img)

        # Write YOLO label file (empty file if no annotations → background image)
        label_lines = []

        if img_filename in gt_grouped.groups:
            rows = gt_grouped.get_group(img_filename)

            for _, row in rows.iterrows():
                x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
                class_id = int(row.class_id)

                # Skip invalid class ids
                if class_id >= NUM_CLASSES:
                    continue

                cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(out_lbl, "w") as f:
            f.write("\n".join(label_lines))

        converted += 1

    print(f"Converted {converted} images to YOLO format.")

    # -- Write dataset.yaml --
    dataset_cfg = {
        "path":  DATASET_DIR.replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    with open(YAML_PATH, "w") as f:
        yaml.dump(dataset_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"Dataset YAML saved to: {YAML_PATH}")

    return True


# -----------------------------
# 4. Generate dataset.yaml only
# -----------------------------

def ensure_yaml():
    """
    If the YOLO dataset already exists but dataset.yaml is missing,
    regenerate it. Useful if the user prepared the dataset manually.
    """
    if os.path.exists(YAML_PATH):
        print(f"dataset.yaml already exists at: {YAML_PATH}")
        return

    if not os.path.isdir(os.path.join(IMAGES_DIR, "train")):
        print(
            "[ERROR] YOLO dataset not found. "
            "Please run build_yolo_dataset() first or prepare the dataset manually."
        )
        return

    dataset_cfg = {
        "path":  DATASET_DIR.replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    with open(YAML_PATH, "w") as f:
        yaml.dump(dataset_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"dataset.yaml regenerated at: {YAML_PATH}")


# -----------------------------
# 5. Train YOLOv8
# -----------------------------

def train_yolo():
    """
    Fine-tune YOLOv8 on the GTSDB dataset.
    Saves the best weights to detection/models/best.pt.
    """

    ensure_yaml()

    if not os.path.exists(YAML_PATH):
        print("[ERROR] Cannot train without dataset.yaml. Exiting.")
        return

    print("\n--- Starting YOLOv8 Training ---")
    print(f"Backbone     : {YOLO_BACKBONE}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Image size   : {IMG_SIZE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Dataset YAML : {YAML_PATH}")

    model = YOLO(YOLO_BACKBONE)

    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name="traffic_sign_detection",
        project=os.path.join(BASE_DIR, "runs"),
        exist_ok=True,
        patience=10,          # early stopping
        save=True,
        val=True,
        plots=True,
    )

    # -------------------------------------------------------
    # Copy best.pt → detection/models/best.pt
    # -------------------------------------------------------
    best_src = os.path.join(
        BASE_DIR, "runs", "traffic_sign_detection", "weights", "best.pt"
    )

    if os.path.exists(best_src):
        best_dst = os.path.join(MODEL_DIR, "best.pt")
        shutil.copy2(best_src, best_dst)
        print(f"\nBest model saved to: {best_dst}")
    else:
        print(
            "\n[WARNING] best.pt not found in expected run directory.\n"
            "Check detection/runs/traffic_sign_detection/weights/ manually."
        )

    return results


# -----------------------------
# 6. Entry point
# -----------------------------

if __name__ == "__main__":

    print("=" * 60)
    print(" YOLOv8 Traffic Sign Detection — Training Script")
    print("=" * 60)

    # Step 1: convert GTSDB → YOLO format (skip if already done)
    converted = build_yolo_dataset()

    if not converted:
        print(
            "\n[INFO] Proceeding with existing dataset in detection/dataset/\n"
            "       Make sure it follows YOLO folder structure."
        )

    # Step 2: train YOLOv8
    train_yolo()

    print("\n[DONE] Training complete.")
