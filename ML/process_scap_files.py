# process_scap_files.py

import os
import sys
from pathlib import Path
import pandas as pd
import logging
import gc
import multiprocessing as mp
from tqdm import tqdm
import joblib 

# Configure logging
logging.basicConfig(filename='output/process_scap_files.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import your preprocessing and SSG modules
from src.preprocessing import extract_syscalls, KNOWN_SYSCALLS, KNOWN_ARGUMENTS
from src.ssg import SystemStateGraph

def process_scap_file(scap_file, output_dir):
    try:
        local_known_syscalls = set()
        local_known_arguments = set()
        features_list = []

        grouped_df = extract_syscalls(scap_file)

        if grouped_df.empty:
            logging.warning(f"No data extracted from {scap_file}")
            return [], local_known_syscalls, local_known_arguments

        local_known_syscalls.update(KNOWN_SYSCALLS)
        local_known_arguments.update(KNOWN_ARGUMENTS)
        KNOWN_SYSCALLS.clear()
        KNOWN_ARGUMENTS.clear()

        ssg = SystemStateGraph()

        for _, row in grouped_df.iterrows():
            syscalls = row['syscall']
            arguments = row['arguments']
            window_graph = ssg.create_window_graph(syscalls, arguments)
            ssg.graph = window_graph
            features = ssg.extract_features()

            features_df = pd.DataFrame([features])

            if 'syscall_node_counts' in features_df.columns:
                syscall_counts_df = pd.json_normalize(features_df['syscall_node_counts']).fillna(0)
                features_df = features_df.drop('syscall_node_counts', axis=1)
                features_df = pd.concat([features_df, syscall_counts_df], axis=1)

            if 'argument_node_counts' in features_df.columns:
                argument_counts_df = pd.json_normalize(features_df['argument_node_counts']).fillna(0)
                features_df = features_df.drop('argument_node_counts', axis=1)
                features_df = pd.concat([features_df, argument_counts_df], axis=1)

            features_df = features_df.fillna(0).astype('float64')
            features_list.append(features_df)

            del features_df
            gc.collect()

        if features_list:
            file_features_df = pd.concat(features_list, ignore_index=True)
            output_filename = f"{scap_file.stem}_features.csv"
            output_file = Path(output_dir) / output_filename
            file_features_df.to_csv(output_file, index=False)

            logging.info(f"Features from {scap_file} saved to {output_file}")
            del file_features_df
            gc.collect()

        del grouped_df
        del ssg
        gc.collect()

        return [output_file], local_known_syscalls, local_known_arguments

    except Exception as e:
        logging.error(f"Error processing {scap_file}: {e}")
        return [], set(), set()

def process_scap_directory(scap_dir, output_dir='output', num_cores=None):
    scap_dir = Path(scap_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scap_dir.is_dir():
        logging.error(f"{scap_dir} is not a valid directory.")
        sys.exit(1)

    scap_files = list(scap_dir.glob('*.scap'))
    if not scap_files:
        logging.error(f"No .scap files found in {scap_dir}")
        sys.exit(1)

    logging.info(f"Processing {len(scap_files)} .scap files...")

    # Use the specified number of cores or default to all available cores
    cores_to_use = num_cores if num_cores is not None else mp.cpu_count()
    logging.info(f"Using {cores_to_use} cores for multiprocessing.")

    with mp.Pool(processes=cores_to_use) as pool:
        results = list(tqdm(pool.starmap(process_scap_file, 
                          [(f, output_dir) for f in scap_files]), 
                          total=len(scap_files), 
                          desc="Processing files"))

    all_output_files = []
    all_known_syscalls = set()
    all_known_arguments = set()

    for output_files, known_syscalls, known_arguments in results:
        all_output_files.extend(output_files)
        all_known_syscalls.update(known_syscalls)
        all_known_arguments.update(known_arguments)

    joblib.dump(all_known_syscalls, Path(output_dir) / 'known_syscalls.pkl')
    joblib.dump(all_known_arguments, Path(output_dir) / 'known_arguments.pkl')
    logging.info("Known syscalls and arguments have been saved.")
    logging.info("Feature extraction completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process .scap files and extract features.")
    parser.add_argument("scap_dir", help="Directory containing .scap files")
    parser.add_argument("--output_dir", "-o", default="output", help="Output directory to save features")
    parser.add_argument("--num_cores", "-c", type=int, help="Number of CPU cores to use for multiprocessing")
    args = parser.parse_args()

    process_scap_directory(args.scap_dir, args.output_dir, args.num_cores)
