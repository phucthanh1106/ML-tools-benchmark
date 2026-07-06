import os
import sys
import argparse
from pathlib import Path

def delete_first_n_pairs(dataset_path, num_files_to_delete=100):
    dataset_dir = Path(dataset_path)
    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"

    # 1. Validation Checks
    if not images_dir.exists() or not labels_dir.exists():
        print(f"[ERROR] Subdirectories not found! Checked:\n -> {images_dir}\n -> {labels_dir}")
        sys.exit(1)

    # 2. Collect and sort the first N images alphabetically
    # Sorting ensures consistent behavior across different OS environments
    all_images = sorted([f for f in os.listdir(images_dir) if not f.startswith('.')])
    images_to_delete = all_images[:num_files_to_delete]

    if not images_to_delete:
        print("[INFO] No images found in the target directory to delete.")
        return

    print(f"[INFO] Found {len(all_images)} total training images.")
    print(f"[PROCESS] Attempting synchronized deletion of the first {len(images_to_delete)} pairs...")

    deleted_images_count = 0
    deleted_labels_count = 0

    # 3. Synchronized Loop Execution
    for img_name in images_to_delete:
        img_file_path = images_dir / img_name
        
        # Determine matching label filename by swapping the extension to .txt
        label_name = f"{img_file_path.stem}.txt"
        label_file_path = labels_dir / label_name

        # Delete Image
        if img_file_path.exists():
            os.remove(img_file_path)
            deleted_images_count += 1

        # Delete Matching Label File
        if label_file_path.exists():
            os.remove(label_file_path)
            deleted_labels_count += 1
        else:
            print(f"[WARNING] Missing expected matching label file for: {img_name}")

    print("\n--- Deletion Summary ---")
    print(f"Successfully deleted images: {deleted_images_count}/{len(images_to_delete)}")
    print(f"Successfully deleted labels: {deleted_labels_count}/{len(images_to_delete)}")
    print("------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronized deletion of matching YOLO image/label pairs.")
    parser.add_argument("--path", type=str, required=True, help="Absolute path to the root dataset folder containing 'train/'")
    args = parser.parse_args()

    delete_first_n_pairs(args.path)