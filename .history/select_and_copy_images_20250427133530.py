import os

# Directory containing the images to rename
data_dir = '/Users/ayush/Desktop/project_ayush_raj/data'

# Get all image files in the directory
image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Sort the files to ensure consistent renaming
image_files.sort()

# Rename the images
for i, image_file in enumerate(image_files, start=1):
    # Construct the new filename
    new_filename = f"image{i}.jpg"
    
    # Get the full paths for the source and destination
    src_path = os.path.join(data_dir, image_file)
    dest_path = os.path.join(data_dir, new_filename)
    
    # Rename the file
    os.rename(src_path, dest_path)

print("Images have been successfully renamed.")