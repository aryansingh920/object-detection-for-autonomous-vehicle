"""
Created on 29/12/2024

@author: Aryan

Filename: __main__.py

Relative Path: src/realtime/__main__.py
"""

import cv2
from ultralytics import YOLO
import os

# If you have a Config class with paths, import it (optional).
# from src.config import Config


def run_realtime_inference(video_source, model_weights, display_window=True):
    """
    Runs real-time (or near real-time) object detection inference on a video source.
    
    :param video_source: Can be a path to a video file or an integer (e.g., 0) for a webcam.
    :param model_weights: Path to the trained YOLO weights (e.g., 'best.pt').
    :param display_window: Whether to display the frames in an OpenCV window.
    """
    # 1. Load the trained YOLO model
    print(f"Loading YOLO model from: {model_weights}")
    model = YOLO(model_weights)

    # 2. Open the video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {video_source}")

    # Optional: set video resolution or FPS if needed
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    print("Starting inference. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch the frame.")
            break

        # 3. Run inference on the frame
        results = model(frame)  # ultralytics >= v8.0.0 syntax
        # results = model.predict(frame)  # older versions might need .predict()

        # 4. Draw bounding boxes using YOLO's built-in plotting
        # results is a list; take the first batch
        annotated_frame = results[0].plot()

        # 5. Display the frame in a window
        if display_window:
            cv2.imshow("Real-Time Inference", annotated_frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit.")
                break

    # 6. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Inference session ended.")


def main():
    """
    Example usage:
      - Replace 'dashcam.mp4' with your dash cam video file
      - Replace 'runs/detect/kitti_experiment/weights/best.pt' with your trained weights
    """
    # If you have a Config with paths, e.g., Config.trained_weights, you can reference it directly.
    # Here, we'll hardcode a path for demonstration.
    video_path = "dashcam.mp4"  # or 0 for webcam
    model_path = "runs/detect/kitti_experiment/weights/best.pt"

    run_realtime_inference(video_source=video_path, model_weights=model_path)


if __name__ == "__main__":
    main()
