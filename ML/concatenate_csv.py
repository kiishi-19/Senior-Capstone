import os
import pandas as pd
import argparse

def concatenate_csv(input_dir, output_file):
    """
    Concatenate all CSV files in the input directory into a single DataFrame
    and save it as a pickle file.

    Args:
        input_dir (str): Directory containing CSV files.
        output_file (str): Path to save the output pickle file.
    """
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    dataframes = []
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Saving combined DataFrame to {output_file}")
    combined_df.to_pickle(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate CSV files into a single pickle file.")
    parser.add_argument("input_dir", type=str, help="Directory containing the CSV files to concatenate.")
    parser.add_argument("output_file", type=str, help="Path to the output pickle file.")
    args = parser.parse_args()

    concatenate_csv(args.input_dir, args.output_file)
