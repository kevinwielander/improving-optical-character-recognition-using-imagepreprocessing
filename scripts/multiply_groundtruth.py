import os
import shutil

folder_path = "../resources/dataset/ground_truth"

for filename in os.listdir(folder_path):
    # Check if the file is a .txt file
    if filename.endswith(".txt"):
        
        base_name = filename[:-4]
        
        # Copy the file 10 times
        for i in range(10):
            # Construct the new filename
            new_filename = f"{base_name}-{str(i).zfill(2)}.txt"
            
            # Get the full path for both files
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Copy the file
            shutil.copy2(old_file_path, new_file_path)

print("Files copied successfully!")
