"""
Created on 03/01/2025

@author: Aryan

Filename: realtime.py
Relative Path: src/realtime/realtime.py

Description:
    This script takes a folder of sequential images, feeds each image (frame) to a 
    trained YOLOv8 model, obtains bounding boxes, and displays the results in a window.
    When played rapidly (e.g., 30 fps), it simulates real-time video inference.

Usage Example:
    python realtime.py --weights models/custom/my_model_final.pt \
                       --images_folder data/images/sequence \
                       --fps 30
"""

import argparse
import os
import time
import cv2
from ultralytics import YOLO


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


def main(weights_path: str, images_folder: str, fps: int):
    """
    Main function for reading sequential images and displaying YOLOv8 real-time inference.
    :param weights_path: Path to the trained YOLOv8 weights file
    :param images_folder: Folder containing sequential images
    :param fps: Frames per second for simulated real-time display
    """
    # 1. Load the YOLOv8 model
    model = YOLO(weights_path)
    print(f"[INFO] Loaded YOLOv8 model from: {weights_path}")

    # 2. Read all image file names from the folder
    #    Sort them by name so that they appear in the correct sequence
    image_files = sorted([
        os.path.join(images_folder, f)
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])

    if not image_files:
        print(f"[ERROR] No images found in {images_folder}")
        return

    print(f"[INFO] Found {len(image_files)} images in folder: {images_folder}")

    # 3. Calculate the delay between frames (in milliseconds) for display
    #    1000 ms / fps
    delay_between_frames = int(1000 / fps)

    # 4. Loop through images repeatedly (you can remove `while True` if you only want one pass)
    while True:
        for img_path in image_files:
            # Read each image
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            # 5. Perform inference (prediction) on the frame
            # By default, returns a list of Results
            results_list = model.predict(frame)
            # For a single image, there's only one Results object
            results = results_list[0]

            # 6. Draw bounding boxes
            drawn_frame = draw_boxes_on_image(frame.copy(), results)

            # 7. Display the frame
            cv2.imshow("YOLOv8 Real-Time Inference", drawn_frame)

            # 8. Wait for the desired frame interval
            #    Press 'q' to quit the loop
            if cv2.waitKey(delay_between_frames) & 0xFF == ord('q'):
                print("[INFO] Quitting real-time display...")
                cv2.destroyAllWindows()
                return

        # If you only want a single pass over the images (no repetition),
        # you can uncomment the following lines:
        # print("[INFO] Finished displaying all images once.")
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv8 inference on a sequence of images.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the trained YOLOv8 weights file (e.g., models/custom/my_model_final.pt).")
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing sequential images.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second to simulate real-time display. Default is 30.")
    args = parser.parse_args()

    main(args.weights, args.images_folder, args.fps)
