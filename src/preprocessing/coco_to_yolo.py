import os
import json


def convert_bbox_coco_to_yolo(image_width, image_height, bbox):
    """
    Convert COCO bounding box format (x_min, y_min, width, height)
    to YOLO format (x_center, y_center, width, height).
    """
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    return [
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    ]


def coco_to_yolo(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ["train", "val", "test"]:
        coco_split_dir = os.path.join(data_dir, split)
        yolo_images_dir = os.path.join(output_dir, split, "images")
        yolo_labels_dir = os.path.join(output_dir, split, "labels")

        if not os.path.exists(yolo_images_dir):
            os.makedirs(yolo_images_dir)
        if not os.path.exists(yolo_labels_dir):
            os.makedirs(yolo_labels_dir)

        # Load image and annotation files
        image_file = os.path.join(coco_split_dir, f"{split}_image.json")
        annotation_file = os.path.join(
            coco_split_dir, f"{split}_annotations.json")

        with open(image_file, "r") as f:
            image_data = json.load(f)

        with open(annotation_file, "r") as f:
            annotation_data = json.load(f)

        # Create a dictionary to map image IDs to their details
        image_id_to_details = {
            img["id"]: img for img in image_data["images"]
        }

        # Create YOLO annotation files
        for annotation in annotation_data["annotations"]:
            image_id = annotation["image_id"]
            category_id = annotation["category_id"] - \
                1  # YOLO categories are 0-indexed
            bbox = annotation["bbox"]

            if image_id not in image_id_to_details:
                continue

            # Get image details
            image_details = image_id_to_details[image_id]
            img_file = os.path.basename(image_details["file_name"])
            img_width = image_details["width"]
            img_height = image_details["height"]

            # YOLO annotation file path
            yolo_annotation_file = os.path.join(
                yolo_labels_dir, os.path.splitext(img_file)[0] + ".txt"
            )

            # Convert COCO bbox to YOLO format
            yolo_bbox = convert_bbox_coco_to_yolo(img_width, img_height, bbox)

            # Write annotations to the YOLO file
            with open(yolo_annotation_file, "a") as yolo_file:
                yolo_bbox_str = " ".join(map(str, yolo_bbox))
                yolo_file.write(f"{category_id} {yolo_bbox_str}\n")

            # Copy the image file to the YOLO images directory
            img_src_path = image_details["file_name"]
            img_dst_path = os.path.join(yolo_images_dir, img_file)
            if not os.path.exists(img_dst_path):
                os.makedirs(os.path.dirname(img_dst_path), exist_ok=True)
                shutil.copy(img_src_path, img_dst_path)
                # os.rename(img_src_path, img_dst_path)


# if __name__ == "__main__":
