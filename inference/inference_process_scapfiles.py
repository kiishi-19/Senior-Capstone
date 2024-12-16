# inference_process_scap_files.py

import os
import sys
from pathlib import Path
import pandas as pd
import logging
import joblib
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import extract_syscalls, KNOWN_SYSCALLS, KNOWN_ARGUMENTS
from src.ssg import SystemStateGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference_pipeline.log")
    ]
)

# Load known syscalls and arguments
def load_known_syscalls_and_arguments(known_syscalls_file, known_arguments_file):
    global KNOWN_SYSCALLS, KNOWN_ARGUMENTS
    KNOWN_SYSCALLS = joblib.load(known_syscalls_file)
    KNOWN_ARGUMENTS = joblib.load(known_arguments_file)
    logging.info(f"Loaded {len(KNOWN_SYSCALLS)} known syscalls and {len(KNOWN_ARGUMENTS)} known arguments.")

def process_scap_file_inference(scap_file, output_dir):
    try:
        features_list = []

        # Extract syscalls, arguments, and timestamps from scap file
        grouped_df = extract_syscalls(scap_file)

        if grouped_df.empty:
            logging.warning(f"No data extracted from {scap_file}")
            return []

        # Initialize SystemStateGraph with the known sets
        ssg = SystemStateGraph(KNOWN_SYSCALLS, KNOWN_ARGUMENTS)

        # Process each time window
        for _, row in grouped_df.iterrows():
            syscalls = row['syscall']
            arguments = row['arguments']
            timestamps = row['timestamp']  # Extract timestamps for inter-arrival time

            # Generate graph and extract features
            window_graph = ssg.create_window_graph(syscalls, arguments, timestamps)
            ssg.graph = window_graph
            features = ssg.extract_features()

            features_df = pd.DataFrame([features])

            # Flatten 'syscall_node_counts' if present
            if 'syscall_node_counts' in features_df.columns:
                syscall_counts_df = pd.json_normalize(features_df['syscall_node_counts']).fillna(0)
                features_df = features_df.drop('syscall_node_counts', axis=1)
                features_df = pd.concat([features_df, syscall_counts_df], axis=1)

            # Flatten 'argument_node_counts' if present
            if 'argument_node_counts' in features_df.columns:
                argument_counts_df = pd.json_normalize(features_df['argument_node_counts']).fillna(0)
                features_df = features_df.drop('argument_node_counts', axis=1)
                features_df = pd.concat([features_df, argument_counts_df], axis=1)

            # Fill NaN values and convert to float
            features_df = features_df.fillna(0).astype('float64')
            features_list.append(features_df)

        # Save features to CSV for analysis
        if features_list:
            file_features_df = pd.concat(features_list, ignore_index=True)
            output_filename = f"{scap_file.stem}_features_inference.csv"
            output_file = Path(output_dir) / output_filename
            file_features_df.to_csv(output_file, index=False)
            logging.info(f"Inference features from {scap_file} saved to {output_file}")

            return [output_file]

        return []

    except Exception as e:
        logging.error(f"Error processing {scap_file} for inference: {e}")
        return []

def process_scap_directory_inference(scap_dir, known_syscalls_file, known_arguments_file, output_dir='inference_output'):
    # Set up directories
    scap_dir = Path(scap_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load known syscalls and arguments from training
    load_known_syscalls_and_arguments(known_syscalls_file, known_arguments_file)

    # Find .scap files for inference
    scap_files = list(scap_dir.glob('*.scap'))
    if not scap_files:
        logging.error(f"No .scap files found in {scap_dir}")
        sys.exit(1)

    logging.info(f"Processing {len(scap_files)} .scap files for inference.")

    # Process each file for inference with progress bar
    all_output_files = []
    for scap_file in tqdm(scap_files, desc="Processing files for inference"):
        output_files = process_scap_file_inference(scap_file, output_dir)
        all_output_files.extend(output_files)

    logging.info("Inference feature extraction completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on .scap files using known syscalls and arguments.")
    parser.add_argument("scap_dir", help="Directory containing .scap files")
    parser.add_argument("known_syscalls_file", help="Path to the pickled known syscalls file")
    parser.add_argument("known_arguments_file", help="Path to the pickled known arguments file")
    parser.add_argument("--output_dir", "-o", default="inference_output", help="Output directory to save inference features")
    args = parser.parse_args()

    # Run inference processing
    process_scap_directory_inference(args.scap_dir, args.known_syscalls_file, args.known_arguments_file, args.output_dir)
