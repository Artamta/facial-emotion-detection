import os

def revert_rename_images(directory):
    """
    Revert the renaming of images back to their original format (e.g., image1.jpg, image2.jpg, ...).

    Args:
        directory (str): Path to the directory containing the renamed images.
    """
    # Get a list of all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    image_files = [f for f in files if f.lower().startswith('img') and f.lower().endswith('.jpg')]

    # Sort the files to ensure consistent ordering
    image_files.sort()

    # Rename each file back to its original format
    for idx, file_name in enumerate(image_files, start=1):
        new_name = f"image{idx}.jpg"  # Format as image1.jpg, image2.jpg, etc.
        old_path = os.path.join(directory, file_name)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Reverted: {file_name} -> {new_name}")

if __name__ == "__main__":
    # Specify the directory containing the renamed images
    image_directory = "./data"  # Update this path if needed
    revert_rename_images(image_directory)