import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO


# -----------------------------------------------------------------------
# evaluate_yolo.py
#
# Evaluates the trained YOLOv8 model.
# Uses Test.csv from GTSRB (Member A's dataset) for ground-truth eval.
#
# Produces:
#   - mAP50 and mAP50-95 (via Ultralytics YOLO validator)
#   - Per-class detection count bar chart
#   - Confidence distribution histogram
#   - Sample detection grid (9 images)
# -----------------------------------------------------------------------


# -----------------------------
# 1. Settings
# -----------------------------

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Dataset paths (auto-detect)
ARCHIVE_DIR  = os.path.join(os.path.dirname(PROJECT_ROOT), "archive_extracted")
ALT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

MODEL_PATH  = os.path.join(BASE_DIR, "models", "best.pt")
YAML_PATH   = os.path.join(BASE_DIR, "dataset", "dataset.yaml")
VAL_IMAGES  = os.path.join(BASE_DIR, "dataset", "images", "val")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.50
IMG_SIZE       = 640

NUM_CLASSES = 43

CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing (vehicles over 3.5t)",
    "Right-of-way at next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5t prohibited", "No entry",
    "General caution", "Dangerous curve (left)", "Dangerous curve (right)",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows (right)",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all restrictions", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing (vehicles over 3.5t)",
]


# -----------------------------
# 2. Locate dataset base
# -----------------------------

def find_archive_base():
    if os.path.isfile(os.path.join(ARCHIVE_DIR, "Test.csv")):
        return ARCHIVE_DIR
    if os.path.isfile(os.path.join(ALT_DATA_DIR, "Test.csv")):
        return ALT_DATA_DIR
    return None


# -----------------------------
# 3. Load model
# -----------------------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run detection/src/train_yolo.py first."
        )
    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
    return model


# -----------------------------
# 4. YOLO validation (mAP)
# -----------------------------

def run_val(model):
    if not os.path.exists(YAML_PATH):
        print(f"[WARNING] dataset.yaml not found at {YAML_PATH}. Skipping mAP eval.")
        return None

    print("\n--- Running YOLO Validation (mAP) ---")

    metrics = model.val(
        data=YAML_PATH,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=True,
    )

    print(f"\nmAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")

    return metrics


# -----------------------------
# 5. Collect inference stats on val images
# -----------------------------

def collect_inference_stats(model, max_images=300):
    """
    Run model on up to max_images val images, collect confidence + class per detection.
    """
    if not os.path.isdir(VAL_IMAGES):
        print(f"[WARNING] Val images not found: {VAL_IMAGES}")
        return []

    img_paths = list(Path(VAL_IMAGES).glob("*.jpg"))[:max_images]

    if not img_paths:
        print("[WARNING] No .jpg images found in val folder.")
        return []

    print(f"\n--- Inference on {len(img_paths)} val images ---")

    all_dets = []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        for box in results[0].boxes:
            cid  = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            all_dets.append({
                "confidence": conf,
                "class_id":   cid,
                "class_name": CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"Class {cid}",
            })

    print(f"Total detections collected: {len(all_dets)}")
    return all_dets


# -----------------------------
# 6. Plot confidence distribution
# -----------------------------

def plot_confidence_distribution(detections):
    if not detections:
        return

    confs = [d["confidence"] for d in detections]

    plt.figure(figsize=(8, 5))
    plt.hist(confs, bins=25, color="steelblue", edgecolor="black")
    plt.title("Detection Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Detections")
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, "conf_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# -----------------------------
# 7. Plot top detected classes
# -----------------------------

def plot_top_classes(detections, top_n=15):
    if not detections:
        return

    from collections import Counter
    counts = Counter(d["class_name"] for d in detections)
    top    = counts.most_common(top_n)

    names  = [t[0] for t in top]
    values = [t[1] for t in top]

    plt.figure(figsize=(10, 7))
    plt.barh(names[::-1], values[::-1], color="steelblue")
    plt.title(f"Top {top_n} Most Detected Traffic Sign Classes")
    plt.xlabel("Number of Detections")
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, "top_classes.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# -----------------------------
# 8. Sample detection grid
# -----------------------------

def save_sample_detections(model, n_samples=9):
    if not os.path.isdir(VAL_IMAGES):
        return

    img_paths = list(Path(VAL_IMAGES).glob("*.jpg"))

    if not img_paths:
        return

    import random
    random.seed(42)
    selected = random.sample(img_paths, min(n_samples, len(img_paths)))

    annotated = []

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        ann = results[0].plot()
        ann = cv2.resize(ann, (320, 240))
        annotated.append(ann)

    if not annotated:
        return

    cols = 3
    rows = (len(annotated) + cols - 1) // cols
    while len(annotated) < rows * cols:
        annotated.append(np.zeros((240, 320, 3), dtype=np.uint8))

    grid = np.vstack([
        np.hstack(annotated[r * cols : (r + 1) * cols])
        for r in range(rows)
    ])

    out = os.path.join(RESULTS_DIR, "sample_detections.jpg")
    cv2.imwrite(out, grid)
    print(f"Sample detections grid saved: {out}")


# -----------------------------
# 9. Summary
# -----------------------------

def print_summary(detections, metrics=None):
    print("\n" + "=" * 60)
    print("      YOLO DETECTION EVALUATION SUMMARY")
    print("=" * 60)

    if metrics:
        print(f"  mAP50          : {metrics.box.map50 * 100:.2f}%")
        print(f"  mAP50-95       : {metrics.box.map * 100:.2f}%")

    if detections:
        print(f"  Sample detections  : {len(detections)}")
        avg_conf = np.mean([d["confidence"] for d in detections])
        print(f"  Avg confidence     : {avg_conf:.4f}")

    print("=" * 60)
    print(f"  Plots  → {PLOTS_DIR}")
    print(f"  Sample → {RESULTS_DIR}/sample_detections.jpg")
    print("=" * 60)


# -----------------------------
# 10. Entry point
# -----------------------------

if __name__ == "__main__":

    print("=" * 60)
    print(" YOLOv8 Traffic Sign Detection — Evaluation")
    print("=" * 60)

    model = load_model()

    metrics    = run_val(model)
    detections = collect_inference_stats(model, max_images=300)

    plot_confidence_distribution(detections)
    plot_top_classes(detections, top_n=15)
    save_sample_detections(model, n_samples=9)

    print_summary(detections, metrics)

    print("\n[DONE] Evaluation complete.")
