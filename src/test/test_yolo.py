"""
Created on 03/01/2025

@author: Aryan

Filename: test_yolo.py
Relative Path: src/test/test_yolo.py
"""
"""
Created on 03/01/2025

@author: Aryan

Filename: test_yolov8.py
Relative Path: src/test/test_yolov8.py
"""




import argparse
import cv2
import os
from ultralytics import YOLO
def draw_boxes_on_image(image, results):
    """
    Draw bounding boxes on the input image using YOLOv8 detection results.
    """

    # YOLOv8 results.boxes contains information like [x1, y1, x2, y2, confidence, class, ...
    for box in results.boxes:
        # box.xyxy[0] returns a tensor of bounding box coords [x1, y1, x2, y2]
        # box.conf, box.cls for confidence and class.
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        # You can map class IDs to class names if you have it
        # For example, if you have a list of class names:
        # class_name = class_list[cls_id]
        # Or you can just use the ID in the bounding box label
        label = f"ID:{cls_id} Conf:{conf:.2f}"

        # Draw rectangle on the image
        color = (0, 255, 0)  # BGR
        thickness = 2
        cv2.rectangle(image, (int(x1), int(y1)),
                      (int(x2), int(y2)), color, thickness)

        # Put label text above the bounding box
        cv2.putText(image,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    color,
                    1,    # line type
                    cv2.LINE_AA)

    return image


def main(weights_path, source, save_dir):
    """
    Main function for testing YOLOv8 model.
    """
    # 1. Load the trained model
    model = YOLO(weights_path)
    print(f"Loaded model from: {weights_path}")

    # 2. Run inference on the source image(s)
    # source can be a single image, folder of images, or video
    # The `model.predict()` returns a list of Results objects (one for each image/video frame)
    results_list = model.predict(source)

    # 3. Draw bounding boxes and save or display the results
    #    results_list[i].orig_img is the original image (numpy array) before resizing/cropping
    for idx, results in enumerate(results_list):
        # Convert the original image to BGR if needed (already in BGR typically if from OpenCV)
        image = results.orig_img.copy()

        # Draw detection results
        drawn_image = draw_boxes_on_image(image, results)

        # Save the image with bounding boxes
        # If source is multiple images, ensure unique naming
        save_path = os.path.join(save_dir, f"pred_{idx}.jpg")
        cv2.imwrite(save_path, drawn_image)
        print(f"Saved result to: {save_path}")

        # If you want to display it instead of saving (optional):
        # cv2.imshow("YOLOv8 Inference", drawn_image)
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a trained YOLOv8 model on images.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the trained YOLOv8 weights file (e.g., models/custom/model_final.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to an image, a folder of images, or a video for inference.")
    parser.add_argument("--save_dir", type=str, default="inference_results",
                        help="Directory where the output images with bounding boxes will be saved.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args.weights, args.source, args.save_dir)
