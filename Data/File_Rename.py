import os

source_dir = "/home/charlie/Desktop/data_csv_files"
i = 1

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        new_filename = f"Dataset_{i}.csv"
        old_file_path = os.path.join(root, file)
        new_file_path = os.path.join(root, new_filename)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} to {new_file_path}")
        i += 1

print("Renaming completed.")
