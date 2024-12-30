import os
import json
import shutil

import yaml


def update_image_paths(json_file_path, images_dir, source_dir):
    """
    Updates the paths of images in the COCO JSON file and copies missing images from a source directory.

    Parameters:
    - json_file_path: str, path to the COCO JSON file.
    - images_dir: str, path to the directory containing images.
    """
    try:
        # Load the JSON file
        print(f"Loading JSON file from {json_file_path}...")
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        updated_count = 0
        skipped_images = []
        copied_images = 0

        # Update image paths
        print("Updating image paths...")
        for image in data['images']:
            # Get the original file name
            original_file_name = image['file_name']
            updated_file_name = os.path.join(
                images_dir, os.path.basename(f"resized_{original_file_name}"))

            if os.path.exists(updated_file_name):
                # If the file exists in the target directory, update the path
                image['file_name'] = updated_file_name
                updated_count += 1
            else:
                # If the file is missing, attempt to copy it from the source directory
                source_file_path = os.path.join(
                    source_dir, os.path.basename(original_file_name))
                if os.path.exists(source_file_path):
                    shutil.copy(source_file_path, updated_file_name)
                    image['file_name'] = updated_file_name
                    copied_images += 1
                else:
                    skipped_images.append(original_file_name)

        # Save the updated JSON file
        updated_json_file_path = json_file_path.replace(
            ".json", "_updated.json")
        with open(updated_json_file_path, 'w') as f:
            json.dump(data, f, indent=4)

        # Print summary
        print(f"Updated {updated_count} image paths.")
        print(
            f"Copied {copied_images} missing images from the source directory.")
        if skipped_images:
            print(f"Skipped {len(skipped_images)} images. Missing files:")
            for img in skipped_images[:10]:  # Show first 10 missing images
                print(f" - {img}")
            print("...")
        else:
            print("All images were successfully updated and/or copied.")

        print(f"Updated JSON saved to {updated_json_file_path}")
        
        with open(updated_json_file_path, "r") as json_file:
            json_data = json.load(json_file)


        # Convert JSON data to YAML
        yaml_data = yaml.dump(json_data, sort_keys=False)
        yaml_file_path = updated_json_file_path.replace(".json", ".yaml")

        # Save YAML data to file
        with open(yaml_file_path, "w") as yaml_file:
            yaml_file.write(yaml_data)

    except Exception as e:
        print(f"Error updating image paths: {e}")


if __name__ == "__main__":
    # Define paths
    json_file_path = r"input/coco/coco_datasets/kitti_coco.json"
    images_dir = r"data/kitti/data_object_image_2/training/image_2"
    # Replace with the actual source directory
    source_dir = r"data/kitti/data_object_image_2/training/image_2"

    # Run the update function
    update_image_paths(json_file_path, images_dir)
