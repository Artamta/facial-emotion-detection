import os
import random
import shutil

def select_and_rename_images(source_root, target_directory, num_images_per_class=10):
    """
    Select a specified number of images randomly from each class directory and save them sequentially.

    Args:
        source_root (str): Path to the root directory containing class-specific subdirectories.
        target_directory (str): Path to the target directory where renamed images will be saved.
        num_images_per_class (int): Number of images to select randomly from each class.
    """
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    current_index = 1  # Start naming images from img01.jpg
    for class_name in os.listdir(source_root):
        class_path = os.path.join(source_root, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip non-directory files

        print(f"[INFO] Processing class: {class_name}")
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Randomly select the specified number of images
        selected_images = random.sample(image_files, min(num_images_per_class, len(image_files)))

        for file_name in selected_images:
            old_path = os.path.join(class_path, file_name)
            new_name = f"img{current_index:02d}.jpg"  # Format as img01.jpg, img02.jpg, etc.
            new_path = os.path.join(target_directory, new_name)
            shutil.copy(old_path, new_path)  # Copy the file to the target directory
            print(f"Copied: {file_name} -> {new_name}")
            current_index += 1

if __name__ == "__main__":
    # Specify the source root directory and target directory
    source_root = "/Users/ayush/Desktop/Fer/Facial-Expression-Recognition.Pytorch/CK+48"
    target_directory = "/Users/ayush/Desktop/project_ayush_raj/data"

    # Select and rename images
    select_and_rename_images(source_root, target_directory, num_images_per_class=10)