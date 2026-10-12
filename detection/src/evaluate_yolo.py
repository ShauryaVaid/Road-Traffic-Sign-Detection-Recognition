import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# -----------------------------------------------------------------------
# evaluate_yolo.py
#
# Evaluates a trained YOLOv8 model on the validation set.
# Produces:
#   - Per-class precision / recall / F1 table (printed to console)
#   - mAP50 and mAP50-95 scores
#   - Saved detection sample grid (detection/results/sample_detections.jpg)
#   - Confidence distribution plot  (detection/results/plots/conf_distribution.png)
# -----------------------------------------------------------------------


# -----------------------------
# 1. Settings
# -----------------------------

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
YAML_PATH  = os.path.join(BASE_DIR, "dataset", "dataset.yaml")
VAL_IMAGES = os.path.join(BASE_DIR, "dataset", "images", "val")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.50
IMG_SIZE       = 640

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
# 2. Load model
# -----------------------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Run detection/src/train_yolo.py first."
        )
    model = YOLO(MODEL_PATH)
    print("Model loaded:", MODEL_PATH)
    return model


# -----------------------------
# 3. Run official YOLO validation
# -----------------------------

def run_val(model):
    """
    Use Ultralytics built-in validator to get mAP and per-class metrics.
    """
    if not os.path.exists(YAML_PATH):
        print(
            f"[WARNING] dataset.yaml not found at {YAML_PATH}.\n"
            "Skipping mAP evaluation. Run train_yolo.py to build the dataset."
        )
        return None

    print("\n--- Running YOLO Validation ---")

    metrics = model.val(
        data=YAML_PATH,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=True,
    )

    print("\n--- Validation Metrics ---")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")

    return metrics


# -----------------------------
# 4. Collect inference stats
# -----------------------------

def collect_inference_stats(model, max_images=200):
    """
    Run inference on up to max_images validation images and collect
    confidence scores and class ids per detection.

    Returns
    -------
    list[dict]  — one dict per detection:
        {"confidence": float, "class_id": int, "class_name": str}
    """
    if not os.path.isdir(VAL_IMAGES):
        print(f"[WARNING] Val images folder not found: {VAL_IMAGES}")
        return []

    img_paths = list(Path(VAL_IMAGES).glob("*.jpg"))[:max_images]

    if not img_paths:
        print("[WARNING] No .jpg images found in val folder.")
        return []

    print(f"\n--- Collecting inference stats on {len(img_paths)} images ---")

    all_detections = []

    for img_path in img_paths:

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        for box in results[0].boxes:
            all_detections.append({
                "confidence": float(box.conf[0].cpu().numpy()),
                "class_id":   int(box.cls[0].cpu().numpy()),
                "class_name": CLASS_NAMES[int(box.cls[0].cpu().numpy())]
                              if int(box.cls[0].cpu().numpy()) < len(CLASS_NAMES)
                              else f"Class {int(box.cls[0].cpu().numpy())}",
            })

    print(f"Total detections collected: {len(all_detections)}")

    return all_detections


# -----------------------------
# 5. Plot confidence distribution
# -----------------------------

def plot_confidence_distribution(detections):
    """
    Plot and save a histogram of confidence scores.
    """
    if not detections:
        print("[INFO] No detections to plot.")
        return

    confidences = [d["confidence"] for d in detections]

    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color="steelblue", edgecolor="black")
    plt.title("Detection Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Detections")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "conf_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Confidence distribution plot saved to: {out_path}")


# -----------------------------
# 6. Plot top-10 detected classes
# -----------------------------

def plot_top_classes(detections, top_n=10):
    """
    Bar chart of the most-frequently detected sign classes.
    """
    if not detections:
        return

    from collections import Counter

    counts = Counter(d["class_name"] for d in detections)
    top    = counts.most_common(top_n)

    names  = [t[0] for t in top]
    values = [t[1] for t in top]

    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], values[::-1], color="steelblue")
    plt.title(f"Top {top_n} Most Detected Traffic Sign Classes")
    plt.xlabel("Number of Detections")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "top_classes.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Top classes plot saved to: {out_path}")


# -----------------------------
# 7. Save detection samples grid
# -----------------------------

def save_sample_detections(model, n_samples=9):
    """
    Run detection on n_samples validation images and save
    an annotated grid image for visual inspection.
    """
    if not os.path.isdir(VAL_IMAGES):
        print(f"[WARNING] Val images not found at {VAL_IMAGES}. Skipping sample grid.")
        return

    img_paths = list(Path(VAL_IMAGES).glob("*.jpg"))

    if not img_paths:
        print("[WARNING] No validation images found.")
        return

    # Limit to n_samples
    import random
    random.seed(42)
    selected = random.sample(img_paths, min(n_samples, len(img_paths)))

    annotated_images = []

    for img_path in selected:

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        annotated = results[0].plot()    # BGR image with boxes drawn by Ultralytics
        annotated = cv2.resize(annotated, (320, 240))
        annotated_images.append(annotated)

    if not annotated_images:
        return

    # Arrange into a grid
    grid_cols = 3
    grid_rows = (len(annotated_images) + grid_cols - 1) // grid_cols

    # Pad to fill grid
    while len(annotated_images) < grid_rows * grid_cols:
        annotated_images.append(
            np.zeros((240, 320, 3), dtype=np.uint8)
        )

    rows_imgs = []
    for r in range(grid_rows):
        row_imgs = annotated_images[r * grid_cols : (r + 1) * grid_cols]
        rows_imgs.append(np.hstack(row_imgs))

    grid = np.vstack(rows_imgs)

    out_path = os.path.join(RESULTS_DIR, "sample_detections.jpg")
    cv2.imwrite(out_path, grid)

    print(f"Sample detection grid saved to: {out_path}")


# -----------------------------
# 8. Print summary stats
# -----------------------------

def print_summary(detections, metrics=None):
    """
    Print a clean summary of evaluation results.
    """
    print("\n" + "=" * 60)
    print("         YOLO DETECTION EVALUATION SUMMARY")
    print("=" * 60)

    if metrics:
        print(f"  mAP50          : {metrics.box.map50:.4f}  ({metrics.box.map50 * 100:.2f}%)")
        print(f"  mAP50-95       : {metrics.box.map:.4f}  ({metrics.box.map * 100:.2f}%)")

    if detections:
        print(f"  Total detections (sample) : {len(detections)}")
        avg_conf = np.mean([d["confidence"] for d in detections])
        print(f"  Average confidence        : {avg_conf:.4f}")

    print("=" * 60)


# -----------------------------
# 9. Entry point
# -----------------------------

if __name__ == "__main__":

    print("=" * 60)
    print(" YOLOv8 Traffic Sign Detection — Evaluation Script")
    print("=" * 60)

    model = load_model()

    # 1. Official YOLO mAP metrics
    metrics = run_val(model)

    # 2. Inference stats on validation images
    detections = collect_inference_stats(model, max_images=200)

    # 3. Plots
    plot_confidence_distribution(detections)
    plot_top_classes(detections, top_n=10)

    # 4. Visual sample grid
    save_sample_detections(model, n_samples=9)

    # 5. Summary
    print_summary(detections, metrics)

    print("\n[DONE] Evaluation complete.")
    print(f"Results saved in: {RESULTS_DIR}")
