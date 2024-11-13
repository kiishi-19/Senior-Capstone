# process_scap_files.py

import os
import sys
from pathlib import Path
from tqdm import tqdm
import joblib
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import your preprocessing and SSG modules
from src.preprocessing import extract_syscalls, KNOWN_SYSCALLS, KNOWN_ARGUMENTS
from src.ssg import SystemStateGraph

def process_scap_directory(scap_dir, output_file=None):
    """
    Process all .scap files in the given directory and extract features.

    Args:
        scap_dir (str or Path): Path to the directory containing .scap files.
        output_file (str or Path, optional): Path to save the extracted features.
    """
    scap_dir = Path(scap_dir)
    if not scap_dir.is_dir():
        print(f"Error: {scap_dir} is not a valid directory.")
        sys.exit(1)
    
    # Find all .scap files in the directory
    scap_files = list(scap_dir.glob('*.scap'))
    if not scap_files:
        print(f"No .scap files found in {scap_dir}")
        sys.exit(1)
    
    feature_list = []
    print(f"Processing {len(scap_files)} .scap files...")
    
    for scap_file in tqdm(scap_files, desc="Processing .scap files"):
        # Step 1: Extract syscalls and arguments
        grouped_df = extract_syscalls(scap_file)
        
        if grouped_df.empty:
            print(f"No data extracted from {scap_file}")
            continue
        
        # Initialize System State Graph
        ssg = SystemStateGraph()
        
        # Process each time window
        for _, row in grouped_df.iterrows():
            syscalls = row['syscall']
            arguments = row['arguments']
            
            # Create window graph
            window_graph = ssg.create_window_graph(syscalls, arguments)
            
            # Update the main graph (optional, if you want to maintain the full graph)
            ssg.graph = window_graph  # For this example, we use the window graph directly
            
            # Extract features from the window graph
            features = ssg.extract_features()
            feature_list.append(features)
    
    # Convert the list of feature dictionaries to a DataFrame
    features_df = pd.DataFrame(feature_list)
    
    # Handle complex features (flatten dictionaries)
    # Flatten 'syscall_node_counts' and 'argument_node_counts' columns
    if 'syscall_node_counts' in features_df.columns:
        syscall_counts_df = pd.json_normalize(features_df['syscall_node_counts']).fillna(0)
        features_df = features_df.drop('syscall_node_counts', axis=1)
        features_df = pd.concat([features_df, syscall_counts_df], axis=1)
    
    if 'argument_node_counts' in features_df.columns:
        argument_counts_df = pd.json_normalize(features_df['argument_node_counts']).fillna(0)
        features_df = features_df.drop('argument_node_counts', axis=1)
        features_df = pd.concat([features_df, argument_counts_df], axis=1)
    
    # Fill missing values with zeros
    features_df = features_df.fillna(0)
    
    # Optionally, save the features to a file
    if output_file:
        output_file = Path(output_file)
        if output_file.suffix == '.csv':
            features_df.to_csv(output_file, index=False)
        elif output_file.suffix == '.parquet':
            features_df.to_parquet(output_file, index=False)
        else:
            print(f"Unsupported output file format: {output_file.suffix}")
            print("Supported formats are .csv and .parquet")
    else:
        print("No output file specified. Features will not be saved.")
    
    # Save the known syscalls and arguments for future use
    joblib.dump(KNOWN_SYSCALLS, 'output/known_syscalls.pkl')
    joblib.dump(KNOWN_ARGUMENTS, 'output/known_arguments.pkl')
    print("Known syscalls and arguments have been saved.")
    
    print("Feature extraction completed.")
    return features_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process .scap files and extract features.")
    parser.add_argument("scap_dir", help="Directory containing .scap files")
    parser.add_argument("--output", "-o", help="Output file to save features (e.g., features.csv or features.parquet)")
    args = parser.parse_args()

    process_scap_directory(args.scap_dir, args.output)
