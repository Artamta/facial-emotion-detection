import os
import random
import shutil

# Source directory containing the 7 emotion folders
source_dir = '/Users/ayush/Desktop/project_ayush_raj/CK+48'

# Destination directory for the selected images
destination_dir = '/Users/ayush/Desktop/project_ayush_raj/data'

# Emotion categories
emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Process each emotion category
for emotion in emotions:
    emotion_dir = os.path.join(source_dir, emotion)  # Path to the emotion folder
    if not os.path.exists(emotion_dir):
        print(f"Warning: Directory {emotion_dir} does not exist. Skipping...")
        continue

    # Get all image files in the emotion directory
    image_files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Randomly select 10 images
    selected_images = random.sample(image_files, 10)

    # Copy and rename the selected images to the destination directory
    for i, image_file in enumerate(selected_images, start=1):
        src_path = os.path.join(emotion_dir, image_file)
        dest_filename = f"{emotion}{i}.jpg"  # Rename the file (e.g., anger1.jpg)
        dest_path = os.path.join(destination_dir, dest_filename)
        shutil.copy(src_path, dest_path)

print("Images have been successfully selected, renamed, and copied to the data folder.")