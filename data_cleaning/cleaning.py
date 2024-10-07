import csv
import os
import re
import random

# Define input and output file paths
input_directory = 'data_txts'
output_file = 'output.csv'

# Function to convert sentiment labels to numerical values
def convert_sentiment(sentiments):
    sentiment_values = []
    sentiment_labels = sentiments.split(',')
    for label in sentiment_labels:
        label = label.strip().lower()
        if label == 'positive':
            sentiment_values.append(1)
        elif label == 'negative':
            sentiment_values.append(-1)
        elif label == 'neutral':
            sentiment_values.append(0)
        else:
            sentiment_values.append('unknown')  # Or use a different value if needed
    return sentiment_values

# Function to remove non-ASCII characters
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

# Separate lists for positive and negative rows
positive_rows = []
negative_rows = []

# Open the output CSV file and write the data
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write header row
    writer.writerow(['Message', 'Sentiment'])
    
    # Loop through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_directory, filename)
            
            # Open the input file and read lines
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
                # Process each line
                for line in lines:
                    # Strip leading/trailing whitespace and remove non-ASCII characters
                    line = line.strip()
                    line = remove_non_ascii(line)
                    
                    # Check for sentiment delimiter and split accordingly
                    if '.@' in line:
                        message, sentiment = line.split('.@', 1)
                    elif '@' in line:
                        message, sentiment = line.split('@', 1)
                    else:
                        message, sentiment = line, 'unknown'
                    
                    sentiment_values = convert_sentiment(sentiment)  # Convert sentiment labels
                    
                    # Add rows to the appropriate list
                    for value in sentiment_values:
                        if value == 1:
                            positive_rows.append((message, value))
                        elif value == -1:
                            negative_rows.append((message, value))

# Determine the number of rows to sample
min_count = min(len(positive_rows), len(negative_rows))

# Sample rows to balance positive and negative sentiments
sampled_positive_rows = random.sample(positive_rows, min_count)
sampled_negative_rows = random.sample(negative_rows, min_count)

# Combine and write to the CSV file
balanced_rows = sampled_positive_rows + sampled_negative_rows

# Shuffle to randomize the order
random.shuffle(balanced_rows)

# Write the balanced rows to the output CSV
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Message', 'Sentiment'])
    writer.writerows(balanced_rows)
