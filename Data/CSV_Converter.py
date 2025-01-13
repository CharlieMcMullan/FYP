import csv
import os
import concurrent.futures

# Define file paths
input_dir = '/home/charlie/Desktop/8GB_Dataset'
output_dir = '/home/charlie/Desktop/data_csv_files'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_file(file_path):
    output_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}.csv"
    output_file_path = os.path.join(output_dir, output_file_name)

    with open(file_path, 'r') as infile, open(output_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        fields_written = False

        for line in infile:
            if line.startswith('#'):
                if line.startswith('#fields') and not fields_written:
                    fields = line.strip().split('\t')[1:]
                    csv_writer.writerow(fields)
                    fields_written = True
                continue

            data = line.strip().split('\t')
            csv_writer.writerow(data)

    print(f"Conversion completed. CSV saved at {output_file_path}")

# Use concurrent processing to handle large file sets efficiently
with concurrent.futures.ProcessPoolExecutor() as executor:
    labeled_files = [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files if file.endswith('.labeled')]
    executor.map(process_file, labeled_files)
