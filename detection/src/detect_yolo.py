import os
import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# 1. Settings
# -----------------------------

# Class names for all 43 GTSRB traffic sign classes
CLASS_NAMES = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing (vehicles over 3.5t)",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve (left)",
    20: "Dangerous curve (right)",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows (right)",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing (vehicles over 3.5t)",
}

# Confidence threshold — detections below this are ignored
CONFIDENCE_THRESHOLD = 0.25

# IoU threshold for non-maximum suppression
IOU_THRESHOLD = 0.45

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")


# -----------------------------
# 2. Load YOLO model
# -----------------------------

def load_model(model_path=MODEL_PATH):
    """
    Load a YOLO model from the given path.

    Parameters
    ----------
    model_path : str
        Path to the trained YOLO weights (.pt file).

    Returns
    -------
    YOLO
        Loaded YOLO model object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Run detection/src/train_yolo.py first to train the model, "
            "or place a pre-trained best.pt in detection/models/."
        )

    model = YOLO(model_path)

    print("YOLO model loaded successfully!")
    print("Model path:", model_path)

    return model


# -----------------------------
# 3. Run detection on an image
# -----------------------------

def detect_signs(model, image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD):
    """
    Run YOLO detection on a single image.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    image : numpy.ndarray
        BGR image read with OpenCV.
    conf : float
        Confidence threshold (0–1).
    iou : float
        IoU threshold for NMS (0–1).

    Returns
    -------
    list[dict]
        Each dict contains:
            - "bbox"       : [x1, y1, x2, y2]  (int pixel coords)
            - "confidence" : float
            - "class_id"   : int
            - "class_name" : str
    """
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        verbose=False
    )

    detections = []

    # results[0] holds detections for the single image
    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0].cpu().numpy())
        class_id   = int(box.cls[0].cpu().numpy())
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")

        detections.append({
            "bbox":        [x1, y1, x2, y2],
            "confidence":  round(confidence, 4),
            "class_id":    class_id,
            "class_name":  class_name,
        })

    return detections


# -----------------------------
# 4. Draw bounding boxes
# -----------------------------

def draw_detections(image, detections):
    """
    Draw bounding boxes and labels on an image.

    Parameters
    ----------
    image : numpy.ndarray
        BGR image.
    detections : list[dict]
        Output from detect_signs().

    Returns
    -------
    numpy.ndarray
        Annotated image (copy of input, not modified in-place).
    """
    annotated = image.copy()

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]

        # Draw bounding box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),   # green
            thickness=2
        )

        # Build label text
        label = f"{class_name} {confidence:.2f}"

        # Choose font scale based on image size
        font_scale = max(0.4, min(annotated.shape[1], annotated.shape[0]) / 1000)
        thickness  = max(1, int(font_scale * 2))

        # Calculate background rectangle for label
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        label_y = max(y1 - 5, text_h + baseline)

        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y + baseline),
            color=(0, 255, 0),
            thickness=cv2.FILLED
        )

        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),    # black text on green background
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    return annotated


# -----------------------------
# 5. Crop detected sign regions
# -----------------------------

def crop_detections(image, detections):
    """
    Crop each detected sign region from the image.

    Parameters
    ----------
    image : numpy.ndarray
        Original BGR image.
    detections : list[dict]
        Output from detect_signs().

    Returns
    -------
    list[numpy.ndarray]
        List of cropped BGR images, one per detection.
    """
    crops = []

    h, w = image.shape[:2]

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        # Clamp to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = image[y1:y2, x1:x2]

        if crop.size > 0:
            crops.append(crop)

    return crops


# -----------------------------
# 6. Run detection on image file
# -----------------------------

def detect_image_file(model, image_path, output_path=None, show=False):
    """
    Run detection on an image file and optionally save/show the result.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    image_path : str
        Path to input image.
    output_path : str or None
        If given, save the annotated image here.
    show : bool
        If True, display the result in a window.

    Returns
    -------
    tuple
        (detections, annotated_image)
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    detections = detect_signs(model, image)

    print(f"\nDetections in '{os.path.basename(image_path)}':")

    if not detections:
        print("  No signs detected.")
    else:
        for i, det in enumerate(detections, 1):
            print(
                f"  [{i}] {det['class_name']} "
                f"(class {det['class_id']}) — "
                f"confidence: {det['confidence']:.4f} — "
                f"bbox: {det['bbox']}"
            )

    annotated = draw_detections(image, detections)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"\nAnnotated image saved to: {output_path}")

    if show:
        cv2.imshow("Detection Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections, annotated


# -----------------------------
# 7. Run detection on video file
# -----------------------------

def detect_video_file(model, video_path, output_path=None, show=False):
    """
    Run detection on every frame of a video file.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    video_path : str
        Path to input video.
    output_path : str or None
        If given, save annotated video here.
    show : bool
        If True, display live while processing.

    Returns
    -------
    int
        Total number of detections across all frames.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {os.path.basename(video_path)}")
    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total}")

    writer = None

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num      = 0
    total_detections = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_num += 1

        detections = detect_signs(model, frame)
        total_detections += len(detections)

        annotated = draw_detections(frame, detections)

        if writer:
            writer.write(annotated)

        if show:
            cv2.imshow("Traffic Sign Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_num % 50 == 0:
            print(f"  Processed frame {frame_num}/{total} ...")

    cap.release()

    if writer:
        writer.release()
        print(f"\nAnnotated video saved to: {output_path}")

    if show:
        cv2.destroyAllWindows()

    print(f"\nTotal detections across {frame_num} frames: {total_detections}")

    return total_detections


# -----------------------------
# 8. Run detection on webcam
# -----------------------------

def detect_webcam(model, camera_index=0):
    """
    Run real-time detection from a webcam feed.
    Press 'q' to quit.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    camera_index : int
        Camera device index (0 for default webcam).
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}")

    print("\nWebcam detection started. Press 'q' to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame from webcam.")
            break

        detections = detect_signs(model, frame)
        annotated  = draw_detections(frame, detections)

        # Show FPS and detection count on frame
        label = f"Signs detected: {len(detections)}"
        cv2.putText(
            annotated,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Traffic Sign Detection — Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")
