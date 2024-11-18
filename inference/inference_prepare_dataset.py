# prepare_inference_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from glob import glob
import logging
import argparse
from joblib import Parallel, delayed

# Define the allowed feature set (same as in training)
ALLOWED_FEATURES = [
    'num_nodes', 'num_edges', 'avg_degree', 'graph_density', 
    'num_connected_components', 'clustering_coefficient',
    'diameter', 'unseen_syscall_influence', 
    'unseen_argument_influence', 'frequency_increase'
]

def ensure_features(chunk):
    """Ensure chunk has all allowed features, filling missing ones with 0, and converting all to numeric."""
    for feature in ALLOWED_FEATURES:
        if feature not in chunk.columns:
            chunk[feature] = 0  # Default fill value for missing features
            logging.warning(f"Feature '{feature}' missing from CSV; filling with 0")
    
    # Ensure 'frequency_increase' is numeric: convert True/False to 1/0
    if 'frequency_increase' in chunk.columns:
        chunk['frequency_increase'] = chunk['frequency_increase'].astype(int)

    # Convert all features to numeric, enforcing consistency
    chunk = chunk[ALLOWED_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    return chunk

def transform_and_save_inference(file, feature_names, scaler, output_dir, chunk_size):
    """Helper function to transform inference data and save scaled features as pickle files."""
    try:
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + ".pkl")
        scaled_data = []

        for chunk in pd.read_csv(file, chunksize=chunk_size):
            # Ensure required features are present and fill missing features
            chunk = ensure_features(chunk)

            # Apply scaling
            scaled_chunk = scaler.transform(chunk)
            scaled_df = pd.DataFrame(scaled_chunk, columns=feature_names)
            scaled_data.append(scaled_df)

        # Concatenate all scaled chunks and save as a pickle file
        final_scaled_df = pd.concat(scaled_data, ignore_index=True)
        final_scaled_df.to_pickle(output_file)
        logging.info(f"Saved scaled data for {file} to {output_file}")

    except Exception as e:
        logging.error(f"Error processing inference file {file}: {e}")

def prepare_inference_data(features_dir, scaler_file='output/scaler.pkl', output_dir='inference_output/scaled_features', chunk_size=10000, n_jobs=-1):
    # Configure logging
    logging.basicConfig(filename='output/prepare_inference_data.log', level=logging.INFO)
    logging.info("Starting inference dataset preparation")
    
    # Validate input directory exists
    if not os.path.isdir(features_dir):
        logging.error(f"Features directory not found: {features_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of CSV files for inference
    csv_files = glob(os.path.join(features_dir, "*.csv"))
    if not csv_files:
        logging.error(f"No CSV files found in directory: {features_dir}")
        return
    
    # Load the pre-trained scaler
    scaler = joblib.load(scaler_file)
    logging.info("Loaded pre-trained scaler.")

    # Set feature names as allowed features
    feature_names = ALLOWED_FEATURES
    
    # Apply scaling and save transformed data as .pkl files in parallel
    Parallel(n_jobs=n_jobs)(delayed(transform_and_save_inference)(file, feature_names, scaler, output_dir, chunk_size) for file in csv_files)
    
    logging.info("Inference data transformation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for inference.")
    parser.add_argument("features_dir", help="Path to the directory containing feature CSV files from inference.")
    parser.add_argument("--scaler_file", default="output/scaler.pkl", help="Path to the pre-trained scaler file.")
    parser.add_argument("--output_dir", default="inference_output/scaled_features", help="Directory to save the scaled inference features as .pkl files.")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Number of rows per chunk when reading CSV files.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs. -1 uses all available cores.")
    args = parser.parse_args()
    
    prepare_inference_data(args.features_dir, args.scaler_file, args.output_dir, chunk_size=args.chunk_size, n_jobs=args.n_jobs)
