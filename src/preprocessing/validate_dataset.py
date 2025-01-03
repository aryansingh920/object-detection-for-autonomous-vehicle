import random
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
        annotation_file = os.path.join(split_dir, f'{self.split}_annotations.json')
        image_file = os.path.join(split_dir, f'{self.split}_image.json')
        calib_file = os.path.join(split_dir, f'{self.split}_calib.json')
        velodyne_file = os.path.join(split_dir, f'{self.split}_velodyne.json')

        # Load annotations, images, calibration, and Velodyne data
        annotations = self.load_data(annotation_file)
        images = self.load_data(image_file)
        calib_data = self.load_data(calib_file)
        velodyne_data = self.load_data(velodyne_file)

        # Create dictionaries by ID for consistency checks
        image_ids = {image['id'] for image in images.get('images', [])}
        calib_ids = {calib['id'] for calib in calib_data['calibration']}
        velodyne_ids = {vel['id'] for vel in velodyne_data['velodyne']}
        annotation_ids = {ann['image_id'] for ann in annotations.get('annotations', [])}

        # Check consistency across all files
        consistent_ids = image_ids & calib_ids & velodyne_ids & annotation_ids
        inconsistent_ids = image_ids ^ calib_ids ^ velodyne_ids ^ annotation_ids

        print(f"Consistent IDs: {len(consistent_ids)}")
        print(f"Inconsistent IDs: {len(inconsistent_ids)}")

        # Identify and save incorrect data
        incorrect_images = [img for img in images.get('images', []) if img['id'] not in consistent_ids]
        incorrect_calib = [calib for calib in calib_data['calibration'] if calib['id'] not in consistent_ids]
        incorrect_velodyne = [vel for vel in velodyne_data['velodyne'] if vel['id'] not in consistent_ids]
        incorrect_annotations = [ann for ann in annotations.get('annotations', []) if
                                 ann['image_id'] not in consistent_ids]

        # Save incorrect data to new files
        incorrect_data = {
            'images': incorrect_images,
            'calibration': incorrect_calib,
            'velodyne': incorrect_velodyne,
            'annotations': incorrect_annotations,
        }

        incorrect_data_file = os.path.join(split_dir, f'{self.split}_incorrect_annotations.json')
        with open(incorrect_data_file, 'w') as f:
            json.dump(incorrect_data, f, indent=4)

        print(f"Incorrect data saved to {incorrect_data_file}")

        # Save validated data for consistent IDs
        validated_data = {
            'images': [img for img in images.get('images', []) if img['id'] in consistent_ids],
            'annotations': [ann for ann in annotations.get('annotations', []) if ann['image_id'] in consistent_ids],
            'categories': annotations.get('categories', []),
        }

        validated_data_file = os.path.join(split_dir, f'{self.split}_validated_annotations.json')
        with open(validated_data_file, 'w') as f:
            json.dump(validated_data, f, indent=4)

        print(f"Validated data saved to {validated_data_file}")

    def save_random_images_with_bboxes(self, split, output_dir, num_images=5):
        """
        Randomly selects images from the dataset, overlays bounding boxes and labels, and saves them.

        Parameters:
        - dataset_path: str, path to the dataset directory.
        - split: str, data split ('train', 'val', or 'test').
        - output_dir: str, directory to save the images with bounding boxes.
        - num_images: int, number of random images to process (default: 5).
        """
        # Paths to input files
        split_dir = os.path.join(self.dataset_path, split)
        annotation_file = os.path.join(split_dir, f'{split}_annotations.json')
        image_file = os.path.join(split_dir, f'{split}_image.json')
        calib_file = os.path.join(split_dir, f'{split}_calib.json')

        # Load data
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        with open(image_file, 'r') as f:
            images = json.load(f)
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)

        # Create mapping for quick lookup
        image_dict = {image['id']: image for image in images.get('images', [])}
        calib_dict = {calib['id']: calib['file_name'] for calib in calib_data['calibration']}

        # Select random images
        selected_images = random.sample(images.get('images', []), min(num_images, len(images.get('images', []))))

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for img_data in selected_images:
            image_id = img_data['id']
            image_path = img_data['file_name']
            calib_file_path = calib_dict.get(image_id, None)

            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist. Skipping.")
                continue

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image {image_path}. Skipping.")
                continue

            # Load P2 matrix from calibration file if available
            P2_matrix = None
            if calib_file_path and os.path.exists(calib_file_path):
                with open(calib_file_path, 'r') as f:
                    calib_lines = f.readlines()
                    for line in calib_lines:
                        if line.startswith('P2'):
                            P2_matrix = np.array([float(x) for x in line.split(':')[1].split()]).reshape(3, 4)
                            break

            has_bbox = False  # Track if any bbox is drawn

            # Overlay bounding boxes
            for ann in annotations.get('annotations', []):
                if ann['image_id'] != image_id:
                    continue

                bbox = ann['bbox']
                if P2_matrix is not None:
                    # Adjust bounding box with calibration
                    x_min, y_min, width, height = bbox
                    x_max, y_max = x_min + width, y_min + height
                    corners = np.array([[x_min, y_min, 1], [x_max, y_min, 1], [x_min, y_max, 1], [x_max, y_max, 1]]).T
                    corners_homogeneous = np.vstack([corners, np.ones((1, corners.shape[1]))])
                    projected = P2_matrix @ corners_homogeneous
                    projected = projected[:2] / projected[2]
                    x_min, y_min = projected.min(axis=1)
                    x_max, y_max = projected.max(axis=1)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                # Draw the bounding box
                x_min, y_min, width, height = map(int, bbox)
                x_max, y_max = x_min + width, y_min + height
                if x_max > x_min and y_max > y_min:  # Valid bbox
                    has_bbox = True
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Add label
                    category_id = ann['category_id']
                    label = next((cat['name'] for cat in annotations.get('categories', []) if cat['id'] == category_id),
                                 "Unknown")
                    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if not has_bbox:
                print(f"No bounding boxes found for image {image_path}. Skipping.")
                continue

            # Save the image
            output_path = os.path.join(output_dir, f"{split}_image_{image_id}.jpg")
            cv2.imwrite(output_path, image)
            print(f"Saved image with bounding boxes to {output_path}")

    def save_random_images_with_bboxes_without_calibration(self, split, output_dir, num_images=5):
        """
        Randomly selects images from the dataset, overlays bounding boxes and labels (without calibration), and saves them.

        Parameters:
        - dataset_path: str, path to the dataset directory.
        - split: str, data split ('train', 'val', or 'test').
        - output_dir: str, directory to save the images with bounding boxes.
        - num_images: int, number of random images to process (default: 5).
        """
        # Paths to input files
        split_dir = os.path.join(self.dataset_path, split)
        annotation_file = os.path.join(split_dir, f'{split}_annotations.json')
        image_file = os.path.join(split_dir, f'{split}_image.json')

        # Load data
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        with open(image_file, 'r') as f:
            images = json.load(f)

        # Create a dictionary for quick image lookup
        image_dict = {image['id']: image for image in images.get('images', [])}

        # Select random images
        selected_images = random.sample(images.get('images', []), min(num_images, len(images.get('images', []))))

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for img_data in selected_images:
            image_id = img_data['id']
            image_path = img_data['file_name']

            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist. Skipping.")
                continue

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image {image_path}. Skipping.")
                continue

            has_bbox = False  # Track if any bbox is drawn

            # Overlay bounding boxes
            for ann in annotations.get('annotations', []):
                if ann['image_id'] != image_id:
                    continue

                bbox = ann['bbox']
                x_min, y_min, width, height = map(int, bbox)
                x_max, y_max = x_min + width, y_min + height

                # Ensure valid bounding box dimensions
                if x_max > x_min and y_max > y_min:
                    has_bbox = True
                    # Draw the bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Add label
                    category_id = ann['category_id']
                    label = next((cat['name'] for cat in annotations.get('categories', []) if cat['id'] == category_id),
                                 "Unknown")
                    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if not has_bbox:
                print(f"No bounding boxes found for image {image_path}. Skipping.")
                continue

            # Save the image
            output_path = os.path.join(output_dir, f"{split}_image_{image_id}.jpg")
            cv2.imwrite(output_path, image)
            print(f"Saved image with bounding boxes to {output_path}")

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
