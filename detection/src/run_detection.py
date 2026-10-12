import os
import sys
import cv2
import argparse
from detect_yolo import load_model, detect_image_file, detect_video_file, detect_webcam


# -----------------------------------------------------------------------
# run_detection.py
#
# Command-line interface for running the YOLO traffic sign detector on:
#   - A single image
#   - A directory of images
#   - A video file
#   - Live webcam feed
#
# Usage examples:
#   python detection/src/run_detection.py --source image --input path/to/img.jpg
#   python detection/src/run_detection.py --source images --input path/to/folder/
#   python detection/src/run_detection.py --source video  --input path/to/video.mp4
#   python detection/src/run_detection.py --source webcam
#   python detection/src/run_detection.py --source image  --input img.jpg --show
#   python detection/src/run_detection.py --source image  --input img.jpg --output out.jpg
# -----------------------------------------------------------------------


BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "output")


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Traffic Sign Detection — Run Script"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["image", "images", "video", "webcam"],
        help="Detection source type.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input image, image folder, or video file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the annotated output. "
             "For 'images' source, this is the output directory.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to trained YOLO model (.pt file). "
             f"Defaults to {MODEL_PATH}.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display result in a window (requires a display / GUI).",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index for webcam source (default: 0).",
    )

    return parser.parse_args()


def run_on_image_folder(model, folder_path, output_dir, show):
    """
    Run detection on all images in a folder and save annotated results.
    """
    import os
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
    img_paths  = [
        p for p in Path(folder_path).iterdir()
        if p.suffix.lower() in extensions
    ]

    if not img_paths:
        print(f"[WARNING] No images found in: {folder_path}")
        return

    print(f"\nFound {len(img_paths)} images in '{folder_path}'")
    total_detections = 0

    for img_path in sorted(img_paths):

        out_path = os.path.join(output_dir, img_path.stem + "_detected.jpg")

        try:
            dets, _ = detect_image_file(
                model,
                str(img_path),
                output_path=out_path,
                show=show,
            )
            total_detections += len(dets)

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")

    print(f"\nProcessed {len(img_paths)} images.")
    print(f"Total detections: {total_detections}")
    print(f"Annotated images saved to: {output_dir}")


def main():

    args = parse_args()

    # Load model
    print("\nLoading YOLO model ...")
    model = load_model(args.model)

    # ---- Image ----
    if args.source == "image":

        if not args.input:
            print("[ERROR] --input is required for source=image.")
            sys.exit(1)

        out_path = args.output
        if out_path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fname    = os.path.basename(args.input)
            out_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + "_detected.jpg")

        detect_image_file(
            model,
            args.input,
            output_path=out_path,
            show=args.show,
        )

    # ---- Image folder ----
    elif args.source == "images":

        if not args.input:
            print("[ERROR] --input (folder path) is required for source=images.")
            sys.exit(1)

        out_dir = args.output if args.output else OUTPUT_DIR

        run_on_image_folder(model, args.input, out_dir, args.show)

    # ---- Video ----
    elif args.source == "video":

        if not args.input:
            print("[ERROR] --input is required for source=video.")
            sys.exit(1)

        out_path = args.output
        if out_path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fname    = os.path.basename(args.input)
            out_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + "_detected.mp4")

        detect_video_file(
            model,
            args.input,
            output_path=out_path,
            show=args.show,
        )

    # ---- Webcam ----
    elif args.source == "webcam":

        detect_webcam(model, camera_index=args.camera)


if __name__ == "__main__":
    main()
