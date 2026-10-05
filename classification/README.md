# Traffic Sign Classification

This module is responsible for the **classification/recognition** part of the Road Traffic Sign Detection and Recognition project.

The goal is to take a cropped traffic sign image and classify it into one of the **43 traffic sign classes** using a Convolutional Neural Network (CNN).

## Project Overview

The complete project consists of multiple components:

* **Detection** – Detects the traffic sign in an image.
* **Classification** – Identifies which traffic sign class the detected sign belongs to.
* **Integration** – Combines detection and classification into a complete system.

This repository contains the **classification module**.

## Dataset

The classification model is trained using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

* Number of classes: **43**
* Training images: **39,209**
* Official test images: **12,630**
* Input image size: **32 × 32 × 3**

The dataset itself is not included in this repository because of its size.

Expected dataset structure:

```text
data/
└── Train/
    ├── 0/
    ├── 1/
    ├── 2/
    ├── ...
    └── 42/
```

## Model Architecture

A CNN-based classification model is used.

```text
Input Image
   ↓
32 × 32 × 3
   ↓
Conv2D (32 filters, ReLU)
   ↓
MaxPooling
   ↓
Conv2D (64 filters, ReLU)
   ↓
MaxPooling
   ↓
Flatten
   ↓
Dense (128 neurons, ReLU)
   ↓
Dropout (0.5)
   ↓
Dense (43 neurons, Softmax)
   ↓
Predicted Traffic Sign Class
```

### Main technologies

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

## Preprocessing

Each input image undergoes the following preprocessing steps:

1. Image is read using OpenCV.
2. Image is resized to **32 × 32 pixels**.
3. Pixel values are normalized from `0–255` to `0–1`.
4. The processed image is passed to the CNN model.

## Train / Validation Split

The training dataset is divided using a stratified split:

* **70%** – Training
* **15%** – Validation
* **15%** – Internal test set

The split is performed using `train_test_split` from Scikit-learn.

## Results

The trained CNN achieved:

| Evaluation              |   Accuracy |
| ----------------------- | ---------: |
| Internal test set       | **99.34%** |
| Official GTSRB test set | **95.20%** |

The official test set contains **12,630 images**.

The official test accuracy is the more important measure of the model's performance because it evaluates the model on the separate GTSRB test dataset.

## Repository Structure

```text
classification/
│
├── models/
│   └── traffic_sign_model.keras
│
├── results/
│   └── plots/
│       ├── accuracy.png
│       └── loss.png
│
└── src/
    ├── preprocessing.py
    ├── data_split.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

### File Description

| File                       | Purpose                                                             |
| -------------------------- | ------------------------------------------------------------------- |
| `preprocessing.py`         | Image resizing and normalization                                    |
| `data_split.py`            | Loads the training dataset and creates train/validation/test splits |
| `train.py`                 | Builds and trains the CNN model                                     |
| `evaluate.py`              | Evaluates the model on the official GTSRB test set                  |
| `predict.py`               | Predicts the class of an individual traffic sign image              |
| `traffic_sign_model.keras` | Trained CNN model                                                   |
| `accuracy.png`             | Training and validation accuracy graph                              |
| `loss.png`                 | Training and validation loss graph                                  |

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ShauryaVaid/Road-Traffic-Sign-Detection-Recognition.git
cd Road-Traffic-Sign-Detection-Recognition
```

### 2. Switch to the classification branch

```bash
git switch feature/classification
```

### 3. Install required libraries

```bash
pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib
```

### 4. Add the dataset

Place the GTSRB training dataset in:

```text
data/Train/
```

with class folders:

```text
0, 1, 2, ..., 42
```

### 5. Train the model

From the project root:

```bash
python classification/src/train.py
```

The trained model will be saved as:

```text
classification/models/traffic_sign_model.keras
```

### 6. Evaluate the model

```bash
python classification/src/evaluate.py
```

This evaluates the trained model on the official GTSRB test dataset.

### 7. Predict an individual image

```bash
python classification/src/predict.py
```

The program asks for the path of a traffic sign image and returns:

```text
Predicted Class: <class_id>
Confidence: <confidence> %
```

## Example Prediction

```text
Prediction:
Predicted Class: 14
Confidence: 100.0 %
```

## Future Improvements

Possible improvements to the classification module include:

* Data augmentation
* Deeper CNN architecture
* Transfer learning
* Hyperparameter tuning
* Handling difficult/low-quality images
* Improving performance on classes with lower recall
* Integration with the traffic sign detection module

## Contribution

This module was developed as part of the **Road Traffic Sign Detection and Recognition** project.

**Contribution:** Traffic Sign Classification using CNN

**Branch:** `feature/classification`
