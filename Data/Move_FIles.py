import os
import shutil
import uuid

# Define source and destination directories
source_dir = "/home/charlie/Desktop/Original_Data/Small/iot_23_datasets_small/opt/Malware-Project/BigDataset/IoTScenarios"
destination_dir = "/home/charlie/Desktop/8GB_Dataset"

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".labeled"):
            # Define full file paths
            source_file = os.path.join(root, file)
            unique_name = f"{uuid.uuid4().hex}_conn.log.labeled"
            destination_file = os.path.join(destination_dir, unique_name)
            
            # Copy file
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {source_file} to {destination_file}")

print("File copying completed.")
