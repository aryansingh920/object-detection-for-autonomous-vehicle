import numpy as np
import json
import os
import cv2


class DataValidator:
    def __init__(self, dataset_path, split, image_size=(640, 640)):
        self.dataset_path = dataset_path
        self.split = split  # train, val, or test
        self.image_size = image_size

    def load_data(self, annotation_file):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations

    def parse_calib_file(self, calib_file_path):
        """Parse the calibration file to extract the P2 matrix."""
        calib = {}
        with open(calib_file_path, 'r') as f:
            for line in f:
                # Skip empty or malformed lines
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                calib[key.strip()] = np.array([float(x)
                                               for x in value.split()])
        if 'P2' not in calib:
            raise ValueError(
                f"P2 matrix not found in calibration file: {calib_file_path}")
        return calib['P2'].reshape(3, 4)

    def project_bbox_with_calib(self, bbox, P2_matrix):
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height

        # Convert to homogeneous coordinates
        corners = np.array([
            [x_min, y_min, 1],
            [x_max, y_min, 1],
            [x_min, y_max, 1],
            [x_max, y_max, 1]
        ]).T  # Shape: (3, 4)

        # Add a row of ones for homogeneous coordinates
        corners_homogeneous = np.vstack(
            [corners, np.ones((1, corners.shape[1]))])  # Shape: (4, 4)
        print("Corners (Homogeneous):", corners_homogeneous)

        # Apply projection using P2_matrix
        projected = P2_matrix @ corners_homogeneous  # Shape: (3, 4)
        print("Projected (Raw):", projected)

        # Normalize by depth (third row of projected matrix)
        projected = projected[:2] / projected[2]  # Shape: (2, 4)
        print("Projected (Normalized):", projected)

        # Extract the min and max X, Y coordinates
        x_min, y_min = projected.min(axis=1)
        x_max, y_max = projected.max(axis=1)

        # Clamp bounding box to image bounds
        x_min = max(0, min(x_min, self.image_size[0]))
        y_min = max(0, min(y_min, self.image_size[1]))
        x_max = max(0, min(x_max, self.image_size[0]))
        y_max = max(0, min(y_max, self.image_size[1]))

        # Ensure valid bounding box dimensions
        if x_max <= x_min or y_max <= y_min:
            print("Invalid Bounding Box after Projection:",
                  [x_min, y_min, x_max, y_max])
            return None

        return [x_min, y_min, x_max, y_max]

    def validate_dataset(self):
        split_dir = os.path.join(self.dataset_path, self.split)
        image_dir = os.path.join(
            self.dataset_path, 'normalized_images', self.split)
        annotation_file = os.path.join(split_dir, f'{self.split}_image.json')
        calib_file = os.path.join(split_dir, f'{self.split}_calib.json')

        # Load annotations and calibration data
        annotations = self.load_data(annotation_file)
        calib_data = self.load_data(calib_file)

        # Create dictionaries for images and calibrations by ID
        image_dict = {image['id']: image for image in annotations.get('images', [])}
        calib_dict = {calib['id']: calib['file_name']
                      for calib in calib_data['calibration']}

        validated_annotations = []

        for annotation in annotations.get('annotations', []):
            image_id = annotation['image_id']

            # Check if image ID exists in both image and calibration dictionaries
            if image_id not in image_dict:
                print(
                    f"Error: Image ID {image_id} not found in 'images'. Skipping annotation.")
                continue
            if image_id not in calib_dict:
                print(
                    f"Error: Calibration data for Image ID {image_id} not found. Skipping annotation.")
                continue

            # Get image and calibration file paths
            image_info = image_dict[image_id]
            image_path = os.path.join(
                image_dir, os.path.basename(image_info['file_name']))
            calib_file_path = calib_dict[image_id]

            # Debug: Print paths
            print(f"Processing Image ID: {image_id}")
            print(f"Image path: {image_path}")
            print(f"Calibration file path: {calib_file_path}")

            # Validate paths
            if not os.path.exists(image_path):
                print(f"Error: Image file {image_path} not found.")
                continue
            if not os.path.exists(calib_file_path):
                print(f"Error: Calibration file {calib_file_path} not found.")
                continue

            # Load image and calibration data
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {image_path}")
                continue

            P2_matrix = self.parse_calib_file(calib_file_path)

            # Adjust bounding box using calibration
            bbox = annotation['bbox']
            adjusted_bbox = self.project_bbox_with_calib(bbox, P2_matrix)

            if adjusted_bbox:
                annotation['bbox'] = adjusted_bbox
                validated_annotations.append(annotation)
            else:
                print(f"Invalid bounding box {bbox} in image ID {image_id}")

            # Visualize bounding boxes
            self.visualize_bbox(
                image.copy(), [adjusted_bbox] if adjusted_bbox else [])

        # Save the validated annotations
        validated_annotations_file = os.path.join(
            split_dir, f'{self.split}_validated_annotations.json')
        with open(validated_annotations_file, 'w') as f:
            json.dump({
                'images': annotations.get('images', []),
                'annotations': validated_annotations,
                'categories': annotations.get('categories', [])
            }, f, indent=4)
        print(f"Validated annotations saved to {validated_annotations_file}")


    def visualize_bbox(self, image, bboxes):
        """Visualize bounding boxes."""
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(image, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow('Image with BBoxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Usage example
if __name__ == "__main__":
    dataset_path = "data/coco"
    validator = DataValidator(dataset_path=dataset_path, split="train")
    validator.validate_dataset()
