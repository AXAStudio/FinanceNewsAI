import pandas as pd

def combine_two_csv(file1, file2, output_file):
    # Read the two CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Concatenate the two DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Output the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f'Combined CSV saved to: {output_file}')

# Example usage
combine_two_csv('old_datasets/split_part1.csv', 'shuffled_main.csv', 'output.csv')
