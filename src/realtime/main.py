import math
import numpy as np
from PIL import Image
import argparse
import os
import cv2
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, ToTensor
import torch
# from src.config.config import Config
CLASS_MAP = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist",
    3: "Truck",
    4: "Van",
    5: "Person_sitting",
    6: "Tram",
    7: "Misc",
}


def pad_to_32(image):
    """Pads the image dimensions to be divisible by 32."""
    height, width, _ = image.shape
    pad_height = math.ceil(height / 32) * 32 - height
    pad_width = math.ceil(width / 32) * 32 - width
    padded_image = cv2.copyMakeBorder(
        image,
        0,
        pad_height,
        0,
        pad_width,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return padded_image, pad_height, pad_width


def load_timestamps(timestamps_path):
    with open(timestamps_path, "r") as f:
        timestamps = [line.strip() for line in f.readlines()]
    print(
        f"[INFO] Loaded {len(timestamps)} timestamps from: {timestamps_path}")
    return timestamps


def overlay_timestamp(image, timestamp, position=(5, 15), font_scale=0.5, color=(0, 0, 0), thickness=1):
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


def load_midas_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    model.eval()
    transform = Compose([
        Resize(384),  # MiDaS input resolution
        ToTensor()
    ])
    print("[INFO] MiDaS model for depth estimation loaded.")
    return model, transform


def depth_estimation(midas_model, transform, image):
    # Convert NumPy array (RGB) to PIL Image
    padded_image, pad_height, pad_width = pad_to_32(image)
    pil_image = Image.fromarray(padded_image)

    # Apply transformations
    input_batch = transform(pil_image).unsqueeze(0)

    # Perform depth estimation
    with torch.no_grad():
        depth = midas_model(input_batch)

    # Remove padding from depth map
    depth = depth.squeeze().numpy()
    depth = depth[:-pad_height, :-pad_width]

    # Normalize depth map for visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    depth_normalized = (depth_normalized * 255).astype("uint8")

    # Resize depth map to match original image size
    depth_normalized = cv2.resize(
        depth_normalized, (image.shape[1], image.shape[0]))

    return depth_normalized

def check_image_sequence(images_folder, timestamps, fps):
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

            if timestamps and i < len(timestamps):
                frame_with_boxes = overlay_timestamp(
                    frame_with_boxes, timestamps[i])

            cv2.imshow("Sequence Check (No Predictions)", frame)

            if cv2.waitKey(delay_between_frames) & 0xFF == ord('q'):
                print("[INFO] Quitting sequence check...")
                cv2.destroyAllWindows()
                return


def realtime_inference(weights_path, images_folder, timestamps, fps, depth_est):
    model = YOLO(weights_path)
    print(f"[INFO] Loaded YOLOv8 model from: {weights_path}")

    midas_model, transform = load_midas_model()  # Load MiDaS

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
    print("[INFO] Displaying real-time predictions and depth estimation (press 'q' to quit).")

    delay_between_frames = int(1000 / fps)

    while True:
        for i, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            # YOLOv8 Prediction
            results_list = model.predict(frame)
            results = results_list[0]
            frame_with_boxes = frame.copy()

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                label = f"{CLASS_MAP.get(cls_id, 'Unknown')} Conf:{conf:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame_with_boxes, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                cv2.putText(frame_with_boxes, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if timestamps and i < len(timestamps):
                frame_with_boxes = overlay_timestamp(
                    frame_with_boxes, timestamps[i])

            # Depth Estimation using MiDaS
            frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)

            if depth_est:
                depth_map = depth_estimation(midas_model, transform, frame_rgb)

                # Stack the frames for display
                combined_frame = cv2.vconcat([frame, frame_with_boxes])
                # Use colormap for better visualization
                depth_colormap = cv2.applyColorMap(
                    depth_map, cv2.COLORMAP_MAGMA)
                final_display = cv2.vconcat([combined_frame, depth_colormap])
                # final_display = cv2.vconcat([combined_frame])
                cv2.imshow("YOLOv8 + Depth Estimation", final_display)

            else:
                # Stack the frames for display
                combined_frame = cv2.vconcat([frame, frame_with_boxes])
                final_display = cv2.vconcat([combined_frame])
                cv2.imshow("YOLOv8", final_display)

            # Display the combined frame

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
    parser.add_argument("--depth_estimation", action="store_true", default=False,
                        help="If set, perform depth estimation using MiDaS.")

    args = parser.parse_args()

    # print("Depth", True if args.depth_estimation else False)
    timestamps = None
    if args.timestamps:
        timestamps = load_timestamps(args.timestamps)

    if args.check_sequence:
        check_image_sequence(args.images_folder, timestamps,
                             args.fps)
    else:
        if not args.weights:
            parser.error(
                "--weights is required unless --check_sequence is specified.")
        realtime_inference(args.weights, args.images_folder,
                           timestamps, args.fps, True if args.depth_estimation else False)
