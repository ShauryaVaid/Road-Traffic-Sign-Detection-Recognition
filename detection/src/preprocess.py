import os
import cv2
import numpy as np


# -----------------------------------------------------------------------
# preprocess.py
#
# Image preprocessing utilities for the detection module.
# These are used internally by detect_yolo.py and may also be used
# by the integration pipeline to prepare frames before feeding them
# to the YOLO model.
# -----------------------------------------------------------------------


# Default target size for YOLO inference
YOLO_IMG_SIZE = 640


# -----------------------------
# 1. Resize image (letterbox)
# -----------------------------

def letterbox(image, target_size=YOLO_IMG_SIZE, color=(114, 114, 114)):
    """
    Resize image to target_size × target_size while preserving aspect
    ratio by padding with a solid colour (letterboxing).

    This matches what YOLOv8's internal preprocessor does, so images
    fed through this function are directly compatible with the model.

    Parameters
    ----------
    image : numpy.ndarray
        BGR image.
    target_size : int
        Target square size (default 640 for YOLOv8).
    color : tuple
        RGB/BGR pad colour. Default is grey (114, 114, 114).

    Returns
    -------
    numpy.ndarray
        Letterboxed image of shape (target_size, target_size, 3).
    """
    h, w = image.shape[:2]

    scale = min(target_size / w, target_size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)

    # Centre the resized image on the canvas
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas


# -----------------------------
# 2. Normalise pixel values
# -----------------------------

def normalize(image):
    """
    Convert a uint8 BGR image to a float32 array normalised to [0, 1].

    Parameters
    ----------
    image : numpy.ndarray  (uint8)
        BGR image.

    Returns
    -------
    numpy.ndarray  (float32)
        Normalised image.
    """
    return image.astype(np.float32) / 255.0


# -----------------------------
# 3. Prepare image for YOLO
# -----------------------------

def prepare_for_yolo(image, target_size=YOLO_IMG_SIZE):
    """
    Full preprocessing pipeline used before feeding a raw image to YOLO.

    Steps:
      1. Letterbox resize to target_size × target_size
      2. Convert BGR → RGB  (YOLO expects RGB)
      3. Normalise to [0, 1]
      4. Add batch dimension → shape (1, target_size, target_size, 3)

    Note: When you call model.predict(source=image, ...) in detect_yolo.py
    with a raw numpy array, Ultralytics handles this preprocessing
    internally. This function is provided for cases where you need
    explicit control (e.g., batch inference or custom pipelines).

    Parameters
    ----------
    image : numpy.ndarray
        Raw BGR image (e.g., from cv2.imread).
    target_size : int
        YOLO input size (default 640).

    Returns
    -------
    numpy.ndarray  (float32, shape (1, H, W, 3))
        Preprocessed image ready for inference.
    """
    # 1. Letterbox
    lb = letterbox(image, target_size=target_size)

    # 2. BGR → RGB
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)

    # 3. Normalise
    norm = normalize(rgb)

    # 4. Add batch dimension
    batch = np.expand_dims(norm, axis=0)

    return batch


# -----------------------------
# 4. Crop a sign region
# -----------------------------

def crop_sign(image, bbox, padding=4):
    """
    Crop a detected sign from the image with optional padding.

    Parameters
    ----------
    image : numpy.ndarray
        Original BGR image.
    bbox : list[int]
        [x1, y1, x2, y2] bounding box coordinates.
    padding : int
        Extra pixels to add around the bounding box (default 4).

    Returns
    -------
    numpy.ndarray or None
        Cropped BGR image, or None if the crop is invalid.
    """
    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox

    # Apply padding and clamp to image bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


# -----------------------------
# 5. Enhance contrast (CLAHE)
# -----------------------------

def enhance_contrast(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    improve visibility in low-contrast or poorly lit images.

    Operates on the L channel in LAB colour space.

    Parameters
    ----------
    image : numpy.ndarray
        BGR image.

    Returns
    -------
    numpy.ndarray
        Contrast-enhanced BGR image.
    """
    lab  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return result
