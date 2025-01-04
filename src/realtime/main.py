"""
Created on 03/01/2025

@author: Aryan

Filename: realtime.py
Relative Path: src/realtime/realtime.py

Description:
    Displays a sequence of images as a video, with optional YOLOv8 predictions.
    If a `timestamps.txt` file is provided, it overlays the timestamp on each frame.

Usage Example:
    # Check the sequence without predictions, with timestamps
    python3.10 src/realtime/main.py \
        --images_folder data/kitti/raw_Data/2011_09_26/2011_09_26_drive_0032_sync/image_00/data \
        --timestamps data/kitti/raw_Data/2011_09_26/2011_09_26_drive_0032_sync/image_00/timestamps.txt \
        --fps 15 \
        --check_sequence

    # Run YOLOv8 inference with timestamps
    python realtime.py \
        --weights models/custom/my_model_final.pt \
        --images_folder data/kitti/raw_Data/2011_09_26/2011_09_26_drive_0032_sync/image_00/data \
        --timestamps data/kitti/raw_Data/2011_09_26/2011_09_26_drive_0032_sync/image_00/timestamps.txt \
        --fps 30
"""

import argparse
import os
import cv2


def load_timestamps(timestamps_path):
    """
    Load timestamps from a file into a list.
    Each line in the file is treated as a separate timestamp.
    """
    with open(timestamps_path, "r") as f:
        timestamps = [line.strip() for line in f.readlines()]
    print(
        f"[INFO] Loaded {len(timestamps)} timestamps from: {timestamps_path}")
    return timestamps


def overlay_timestamp(image, timestamp, position=(5, 15), font_scale=0.5, color=(0, 0, 0), thickness=1):
    """
    Overlay a timestamp on the given image.
    :param image: Input image (numpy array).
    :param timestamp: String timestamp to overlay.
    :param position: Position (x, y) for the text.
    :param font_scale: Font size.
    :param color: Text color in BGR format.
    :param thickness: Thickness of the text.
    :return: Image with timestamp overlaid.
    """
    cv2.putText(
        image,
        timestamp,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    return image


def check_image_sequence(images_folder, timestamps, fps):
    """
    Display images with timestamps in a folder without predictions.
    """
    image_files = sorted([
        os.path.join(images_folder, f)
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])

    if not image_files:
        print(f"[ERROR] No images found in {images_folder}")
        return

    if timestamps and len(image_files) != len(timestamps):
        print(
            f"[WARN] Number of images ({len(image_files)}) does not match number of timestamps ({len(timestamps)}).")

    print(f"[INFO] Found {len(image_files)} images in folder: {images_folder}")
    print("[INFO] Displaying images WITHOUT predictions (press 'q' to quit).")

    delay_between_frames = int(1000 / fps)

    while True:
        for i, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            # Overlay timestamp if available
            if timestamps and i < len(timestamps):
                frame = overlay_timestamp(frame, timestamps[i])

            cv2.imshow("Sequence Check (No Predictions)", frame)

            if cv2.waitKey(delay_between_frames) & 0xFF == ord('q'):
                print("[INFO] Quitting sequence check...")
                cv2.destroyAllWindows()
                return


def realtime_inference(weights_path, images_folder, timestamps, fps):
    """
    Run YOLOv8 inference on a folder of sequential images with optional timestamps.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    print(f"[INFO] Loaded YOLOv8 model from: {weights_path}")

    image_files = sorted([
        os.path.join(images_folder, f)
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])

    if not image_files:
        print(f"[ERROR] No images found in {images_folder}")
        return

    if timestamps and len(image_files) != len(timestamps):
        print(
            f"[WARN] Number of images ({len(image_files)}) does not match number of timestamps ({len(timestamps)}).")

    print(f"[INFO] Found {len(image_files)} images in folder: {images_folder}")
    print("[INFO] Displaying real-time predictions (press 'q' to quit).")

    delay_between_frames = int(1000 / fps)

    while True:
        for i, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            # Perform inference
            results_list = model.predict(frame)
            results = results_list[0]

            # Draw bounding boxes
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                label = f"ID:{cls_id} Conf:{conf:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Overlay timestamp if available
            if timestamps and i < len(timestamps):
                frame = overlay_timestamp(frame, timestamps[i])

            # Display the frame
            cv2.imshow("YOLOv8 Real-Time Inference", frame)

            if cv2.waitKey(delay_between_frames) & 0xFF == ord('q'):
                print("[INFO] Quitting real-time display...")
                cv2.destroyAllWindows()
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate real-time YOLOv8 inference or just check image sequence.")
    parser.add_argument("--weights", type=str,
                        help="Path to the trained YOLOv8 weights file (e.g., models/custom/my_model_final.pt).")
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing sequential images.")
    parser.add_argument("--timestamps", type=str,
                        help="Path to the timestamps file (optional).")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second to simulate display. Default=30.")
    parser.add_argument("--check_sequence", action="store_true",
                        help="If set, only check the image sequence without running predictions.")

    args = parser.parse_args()

    # Load timestamps if provided
    timestamps = None
    if args.timestamps:
        timestamps = load_timestamps(args.timestamps)

    if args.check_sequence:
        check_image_sequence(args.images_folder, timestamps, args.fps)
    else:
        if not args.weights:
            parser.error(
                "--weights is required unless --check_sequence is specified.")
        realtime_inference(args.weights, args.images_folder,
                           timestamps, args.fps)
