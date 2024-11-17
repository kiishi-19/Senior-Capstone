from joblib import Parallel, delayed
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from glob import glob
import logging
import argparse

# Define the allowed feature set
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

def transform_and_save(file, feature_names, scaler, output_dir, chunk_size):
    """Helper function to transform data and save scaled features."""
    try:
        output_file = os.path.join(output_dir, os.path.basename(file))
        first_chunk = True
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            # Ensure required features are present and fill missing features
            chunk = ensure_features(chunk)

            # Apply scaling
            scaled_chunk = scaler.transform(chunk)
            scaled_df = pd.DataFrame(scaled_chunk, columns=feature_names)

            # Log mean and std for each column to confirm scaling
            means = scaled_df.mean().round(4)
            std_devs = scaled_df.std().round(4)
            logging.info(f"File: {file}, Mean values after scaling:\n{means}")
            logging.info(f"File: {file}, Std deviation after scaling:\n{std_devs}")

            # Save scaled data
            scaled_df.to_csv(output_file, index=False, mode='w' if first_chunk else 'a', header=first_chunk)
            first_chunk = False
    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")

def prepare_dataset(features_dir, scaler_file='output/scaler.pkl', output_dir='output/scaled_features', chunk_size=10000, n_jobs=-1):
    # Configure logging
    logging.basicConfig(filename='output/prepare_dataset.log', level=logging.INFO)
    logging.info("Starting dataset preparation")
    
    # Validate input directory exists
    if not os.path.isdir(features_dir):
        logging.error(f"Features directory not found: {features_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of CSV files
    csv_files = glob(os.path.join(features_dir, "*.csv"))
    if not csv_files:
        logging.error(f"No CSV files found in directory: {features_dir}")
        return
    
    # Set feature names as allowed features
    feature_names = ALLOWED_FEATURES
    scaler = StandardScaler()
    
    # Step 1: Fit the scaler on numeric columns of the allowed features
    for file in csv_files:
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            # Ensure required features are present and fill missing features
            chunk = ensure_features(chunk)
            scaler.partial_fit(chunk)
    
    joblib.dump(scaler, scaler_file)
    
    # Step 2: Parallelize the data scaling process
    Parallel(n_jobs=n_jobs)(delayed(transform_and_save)(file, feature_names, scaler, output_dir, chunk_size) for file in csv_files)
    
    logging.info("Data transformation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("features_dir", help="Path to the directory containing feature CSV files.")
    parser.add_argument("--scaler_file", default="output/scaler.pkl", help="File to save the scaler.")
    parser.add_argument("--output_dir", default="output/scaled_features", help="Directory to save the scaled features.")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Number of rows per chunk when reading CSV files.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs. -1 uses all available cores.")
    args = parser.parse_args()
    
    prepare_dataset(args.features_dir, args.scaler_file, args.output_dir, chunk_size=args.chunk_size, n_jobs=args.n_jobs)
