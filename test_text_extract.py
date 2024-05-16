import os

# Categories to keep
categories_to_keep = ["02691156", "02828884"]
dir_path = "/root/ODGNet/data/ShapeNet55-34/ShapeNet-55"
# Open the text file for reading
with open(f"{dir_path}/test_copy.txt", "r") as file:
    lines = file.readlines()

# Open a temporary file for writing
with open("temp_file.txt", "w") as temp_file:
    for line in lines:
        # Extract the category from the line
        category = line.split("-")[0]

        # Check if the category is in the list of categories to keep
        if category in categories_to_keep:
            temp_file.write(line)

# Open the input file for writing (this will overwrite the file)
with open(f"{dir_path}/test_copy.txt", "w") as file:
    with open("temp_file.txt", "r") as temp_file:
        file.writelines(temp_file.readlines())

# Remove the temporary file
os.remove("temp_file.txt")