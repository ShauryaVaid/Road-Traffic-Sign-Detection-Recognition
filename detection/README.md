# Traffic Sign Detection — YOLO Module

This module is responsible for the **detection** part of the Road Traffic Sign Detection and Recognition project.

The goal is to **locate traffic signs** in a full image or video frame (bounding boxes) using a fine-tuned **YOLOv8** model trained on the **German Traffic Sign Detection Benchmark (GTSDB)**.

---

## Project Overview

The complete project has four modules:

| Module | Branch | Responsibility |
|---|---|---|
| **Data Preprocessing** | `feature/data-preprocessing` | Dataset cleaning, augmentation, train/val/test split |
| **Detection** ← *this module* | `feature/detection` | Localise signs → bounding boxes |
| **Classification** | `feature/classification` | Identify which sign class it is |
| **Integration** | `feature/integration` | End-to-end pipeline |

---

## Dataset

Training uses the **German Traffic Sign Detection Benchmark (GTSDB)**.

| Property | Value |
|---|---|
| Images | 900 full road-scene images |
| Sign classes | 43 (same as GTSRB) |
| Annotation format | `gt.txt` — `filename;x1;y1;x2;y2;class_id` |
| Image format | `.ppm` |

The dataset is **not included** in this repository due to size.

**Expected raw data structure:**

```text
data/
└── gtsdb/
    ├── Images/          ← all .ppm files (00000.ppm … 00899.ppm)
    └── gt.txt           ← ground-truth annotations
```

> Download GTSDB from: https://benchmark.ini.rub.de/gtsdb_news.html

---

## Model

**YOLOv8n** (nano) fine-tuned on GTSDB.

| Property | Value |
|---|---|
| Backbone | YOLOv8n (pre-trained on COCO) |
| Input size | 640 × 640 |
| Classes | 43 traffic sign types |
| Epochs | 50 (with early stopping) |
| Confidence threshold | 0.25 |
| IoU threshold | 0.45 |

The trained model weights are stored at:

```text
detection/models/best.pt
```

---

## Results

The trained YOLOv8n detector achieved the following benchmarks on the validation dataset (5,882 images):

| Metric | Score |
|---|---|
| **mAP@50** | **85.0%** (0.850) |
| **mAP@50-95** | **73.3%** (0.733) |
| **Precision** | **81.2%** (0.812) |
| **Recall** | **77.9%** (0.779) |
| **Inference Latency** | **~11.6 ms / frame** (Real-Time capable) |

Evaluation graphs and visual samples are generated in:
* `detection/results/plots/conf_distribution.png` (Detection confidence distribution)
* `detection/results/plots/top_classes.png` (Most frequently detected sign classes)
* `detection/results/sample_detections.jpg` (Visual 3x3 sample detection grid)

---

## Repository Structure

```text
detection/
│
├── src/
│   ├── detect_yolo.py       ← Core detection API (load, detect, draw, crop)
│   ├── train_yolo.py        ← GTSDB → YOLO conversion + YOLOv8 training
│   ├── evaluate_yolo.py     ← mAP, confidence plots, sample grid
│   ├── run_detection.py     ← CLI runner (image / folder / video / webcam)
│   └── preprocess.py        ← Preprocessing utilities (letterbox, CLAHE, crop)
│
├── dataset/
│   ├── images/
│   │   ├── train/           ← Training images (.jpg)
│   │   └── val/             ← Validation images (.jpg)
│   ├── labels/
│   │   ├── train/           ← YOLO label .txt files
│   │   └── val/
│   └── dataset.yaml         ← YOLO dataset configuration
│
├── models/
│   └── best.pt              ← Trained YOLO weights (generated after training)
│
├── results/
│   ├── plots/
│   │   ├── conf_distribution.png
│   │   └── top_classes.png
│   ├── output/              ← Annotated images / videos from run_detection.py
│   └── sample_detections.jpg
│
├── runs/                    ← Ultralytics training run logs
└── requirements.txt
```

### File Descriptions

| File | Purpose |
|---|---|
| `detect_yolo.py` | Core module. Loads model, runs inference, draws bounding boxes, crops sign regions. Used by integration module. |
| `train_yolo.py` | Converts GTSDB to YOLO format, writes `dataset.yaml`, fine-tunes YOLOv8. |
| `evaluate_yolo.py` | Evaluates trained model: mAP metrics, confidence distribution, top-class chart, sample detection grid. |
| `run_detection.py` | CLI interface for running detection on image / image folder / video / webcam. |
| `preprocess.py` | Shared utilities: letterbox resize, normalisation, CLAHE contrast enhancement, sign cropping. |

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ShauryaVaid/Road-Traffic-Sign-Detection-Recognition.git
cd Road-Traffic-Sign-Detection-Recognition
git checkout feature/detection
```

### 2. Install dependencies

```bash
pip install -r detection/requirements.txt
```

### 3. Add the GTSDB dataset

Place the GTSDB data at:

```text
data/
└── gtsdb/
    ├── Images/
    └── gt.txt
```

### 4. Train the model

```bash
python detection/src/train_yolo.py
```

This will:
1. Convert GTSDB annotations to YOLO format
2. Create `detection/dataset/` with `images/` and `labels/`
3. Write `detection/dataset/dataset.yaml`
4. Fine-tune YOLOv8n for 50 epochs
5. Save `detection/models/best.pt`

### 5. Evaluate the model

```bash
python detection/src/evaluate_yolo.py
```

Outputs:
- `mAP50` and `mAP50-95` printed to console
- `detection/results/plots/conf_distribution.png`
- `detection/results/plots/top_classes.png`
- `detection/results/sample_detections.jpg`

### 6. Run detection

**On a single image:**

```bash
python detection/src/run_detection.py --source image --input path/to/image.jpg --show
```

**On a folder of images:**

```bash
python detection/src/run_detection.py --source images --input path/to/folder/ --output detection/results/output/
```

**On a video:**

```bash
python detection/src/run_detection.py --source video --input path/to/video.mp4 --show
```

**Live webcam:**

```bash
python detection/src/run_detection.py --source webcam
```

> Press **q** to quit the webcam / video window.

---

## Integration with Classification Module

The `detect_yolo.py` module exposes these functions that the integration module (Member D) connects to the classification module (Member C):

```python
from detection.src.detect_yolo import load_model, detect_signs, crop_detections

# Load YOLO model
model = load_model("detection/models/best.pt")

# Detect signs in an image (returns list of dicts with bbox, confidence, class_id)
detections = detect_signs(model, image)

# Crop each detected sign region (for feeding to CNN classifier)
crops = crop_detections(image, detections)
```

**Output format of `detect_signs()`:**

```python
[
    {
        "bbox":        [x1, y1, x2, y2],   # pixel coordinates
        "confidence":  0.93,
        "class_id":    14,
        "class_name":  "Stop",
    },
    ...
]
```

---

## 43 Traffic Sign Classes

| Class ID | Name |
|---|---|
| 0 | Speed limit (20km/h) |
| 1 | Speed limit (30km/h) |
| 2 | Speed limit (50km/h) |
| 3 | Speed limit (60km/h) |
| 4 | Speed limit (70km/h) |
| 5 | Speed limit (80km/h) |
| 6 | End of speed limit (80km/h) |
| 7 | Speed limit (100km/h) |
| 8 | Speed limit (120km/h) |
| 9 | No passing |
| 10 | No passing (vehicles over 3.5t) |
| 11 | Right-of-way at next intersection |
| 12 | Priority road |
| 13 | Yield |
| 14 | Stop |
| 15 | No vehicles |
| 16 | Vehicles over 3.5t prohibited |
| 17 | No entry |
| 18 | General caution |
| 19 | Dangerous curve (left) |
| 20 | Dangerous curve (right) |
| 21 | Double curve |
| 22 | Bumpy road |
| 23 | Slippery road |
| 24 | Road narrows (right) |
| 25 | Road work |
| 26 | Traffic signals |
| 27 | Pedestrians |
| 28 | Children crossing |
| 29 | Bicycles crossing |
| 30 | Beware of ice/snow |
| 31 | Wild animals crossing |
| 32 | End of all restrictions |
| 33 | Turn right ahead |
| 34 | Turn left ahead |
| 35 | Ahead only |
| 36 | Go straight or right |
| 37 | Go straight or left |
| 38 | Keep right |
| 39 | Keep left |
| 40 | Roundabout mandatory |
| 41 | End of no passing |
| 42 | End of no passing (vehicles over 3.5t) |

---

## Contribution

This module was developed as part of the **Road Traffic Sign Detection and Recognition** project.

**Contribution:** Traffic Sign Detection using YOLOv8

**Branch:** `feature/detection`
