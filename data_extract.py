import os
import shutil

# Set the source directory and target directory
src_dir = "/root/ODGNet/data/ShapeNet55-34/shapenet_pc"
target_dir = "/root/ODGNet/data/ShapeNet55-34/shapenet_subset"

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Categories to filter
categories = ["02691156", "02828884"]

# Iterate over files in the source directory
for root, dirs, files in os.walk(src_dir):
    for file in files:
        # Extract the category from the file name
        category = file.split("-")[0]

        # Check if the category matches the desired categories
        if category in categories:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_dir, file)

            # Move the file to the target directory
            shutil.move(src_file, dst_file)
            print(f"Moved {file} to {target_dir}")