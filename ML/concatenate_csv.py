import pandas as pd
import os

# Define the directory containing CSV files and the output file path
input_dir = '/home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features'
output_file = '/home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features.pkl'

# Read and concatenate all CSV files in the directory
dataframes = []
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)
        dataframes.append(df)

# Concatenate all dataframes
concatenated_df = pd.concat(dataframes, ignore_index=True)

# Save the concatenated dataframe as a pickle file
concatenated_df.to_pickle(output_file)

print(f'Concatenated dataframe saved as {output_file}')
