"""
Created on 03/01/2025

@author: Aryan

Filename: realtime.py
Relative Path: src/realtime/realtime.py

Description:
    1) Repeatedly display a folder of sequential images in real-time,
       optionally running YOLOv8 inference to draw bounding boxes.
    2) Includes a separate function to just check the image sequence 
       without applying YOLO predictions.

Usage Example:
    # 1) Check the image sequence only (no predictions)
    python realtime.py \
        --images_folder data/images/sequence \
        --fps 30 \
        --check_sequence

    # 2) Run YOLOv8 real-time inference on the folder of images
    python realtime.py \
        --weights models/custom/my_model_final.pt \
        --images_folder data/images/sequence \
        --fps 30
"""

import argparse
import os
import time
import cv2

# Only import YOLO if we need it
# (We can do a lazy import inside main if we skip check_sequence)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def draw_boxes_on_image(image, results):
    """
    Draw bounding boxes on the input image using YOLOv8 detection results.
    """
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        label = f"ID:{cls_id} Conf:{conf:.2f}"

        color = (0, 255, 0)  # BGR format
        thickness = 2
        cv2.rectangle(image, (int(x1), int(y1)),
                      (int(x2), int(y2)), color, thickness)
        cv2.putText(
            image,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
    return image


def check_image_sequence(images_folder, fps):
    """
    Simply display images in a folder in sequence, simulating real-time,
    WITHOUT running YOLO predictions. Press 'q' to exit.
    """
    image_files = sorted([
        os.path.join(images_folder, f)
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])

    if not image_files:
        print(f"[ERROR] No images found in {images_folder}")
        return

    print(f"[INFO] Found {len(image_files)} images in folder: {images_folder}")
    print("[INFO] Displaying images WITHOUT predictions (press 'q' to quit).")

    delay_between_frames = int(1000 / fps)

    # Loop through the images in a loop (you can remove while True for single pass)
    while True:
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            cv2.imshow("Sequence Check (No Predictions)", frame)

            if cv2.waitKey(delay_between_frames) & 0xFF == ord('q'):
                print("[INFO] Quitting sequence check...")
                cv2.destroyAllWindows()
                return

        # Uncomment below lines if only one pass is desired:
        # print("[INFO] Finished displaying all images once.")
        # break


def realtime_inference(weights_path, images_folder, fps):
    """
    Run YOLOv8 inference on a folder of sequential images, displaying
    bounding boxes in real-time. Press 'q' to quit.
    """
    if YOLO is None:
        raise ImportError(
            "Ultralytics not installed or import failed. "
            "Please install via `pip install ultralytics`."
        )

    # 1. Load the YOLOv8 model
    model = YOLO(weights_path)
    print(f"[INFO] Loaded YOLOv8 model from: {weights_path}")

    # 2. Read image file names
    image_files = sorted([
        os.path.join(images_folder, f)
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])

    if not image_files:
        print(f"[ERROR] No images found in {images_folder}")
        return

    print(f"[INFO] Found {len(image_files)} images in folder: {images_folder}")
    print("[INFO] Displaying real-time predictions (press 'q' to quit).")

    delay_between_frames = int(1000 / fps)

    # 3. Loop through images repeatedly
    while True:
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            # 4. Perform inference on the frame
            results_list = model.predict(frame)
            results = results_list[0]  # single image => one Results object

            # 5. Draw bounding boxes
            drawn_frame = draw_boxes_on_image(frame.copy(), results)

            # 6. Display the frame
            cv2.imshow("YOLOv8 Real-Time Inference", drawn_frame)

            if cv2.waitKey(delay_between_frames) & 0xFF == ord('q'):
                print("[INFO] Quitting real-time display...")
                cv2.destroyAllWindows()
                return

        # If only one pass is desired, uncomment:
        # print("[INFO] Finished displaying all images once.")
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate real-time YOLOv8 inference or just check image sequence.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to the trained YOLOv8 weights file (e.g., models/custom/my_model_final.pt).")
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing sequential images.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second to simulate display. Default=30.")
    parser.add_argument("--check_sequence", action="store_true",
                        help="If set, only check the image sequence without running predictions.")

    args = parser.parse_args()

    if args.check_sequence:
        # Just display the images without predictions
        check_image_sequence(args.images_folder, args.fps)
    else:
        # Run real-time YOLOv8 inference if weights are provided
        if not args.weights:
            parser.error(
                "--weights is required unless --check_sequence is specified.")
        realtime_inference(args.weights, args.images_folder, args.fps)
