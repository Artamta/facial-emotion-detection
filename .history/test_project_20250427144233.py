import os

def delete_images(directory):
    """
    Delete all images in the specified directory.

    Args:
        directory (str): Path to the directory containing the images.
    """
    # Get a list of all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Delete each image file
    for file_name in image_files:
        file_path = os.path.join(directory, file_name)
        os.remove(file_path)
        print(f"Deleted: {file_name}")

if __name__ == "__main__":
    # Specify the directory containing the images
    image_directory = "./data"  # Update this path if needed
    delete_images(image_directory)