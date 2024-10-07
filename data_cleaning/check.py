import csv
import re

# Define the input and output file paths
input_file1 = 'output.csv'  # First CSV file
input_file2 = 'shuffled_main.csv'  # Second CSV file
output_file = 'merged_output.csv'  # Output CSV file

# Function to remove non-ASCII characters from a string
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

# Function to read CSV into a list of rows with non-ASCII characters removed
def read_and_clean_csv(file_path):
    cleaned_rows = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            cleaned_row = [remove_non_ascii(cell) for cell in row]
            cleaned_rows.append(cleaned_row)
    return cleaned_rows

# Read and clean both CSV files
rows1 = read_and_clean_csv(input_file1)
rows2 = read_and_clean_csv(input_file2)

# Check if the header is present in both files and assume the headers are the same
header1 = rows1[0]
header2 = rows2[0]

# Combine rows, skipping the header from the second file if the headers are the same
combined_rows = rows1 + rows2[1:] if header1 == header2 else rows1 + rows2

# Write the combined data to the output CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(combined_rows)

print(f'CSV files have been merged and saved to {output_file}')
