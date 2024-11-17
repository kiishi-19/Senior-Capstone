# prepare_dataset.py

import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import logging
import gc
from glob import glob

def prepare_dataset(features_dir, scaler_file='output/scaler.pkl', output_dir='output/scaled_features', chunk_size=10000):
    """
    Load features from CSV files in a directory, scale them incrementally, and save the scaler and scaled features.

    Args:
        features_dir (str): Path to the directory containing feature CSV files.
        scaler_file (str): Path to save the scaler.
        output_dir (str): Path to save the scaled features.
        chunk_size (int): Number of rows per chunk when reading CSV files.
    """
    try:
        # Configure logging
        logging.basicConfig(filename='output/prepare_dataset.log', level=logging.INFO,
                            format='%(asctime)s %(message)s')
        logging.info("Starting dataset preparation")
        
        # Validate input directory exists
        if not os.path.isdir(features_dir):
            logging.error(f"Features directory not found: {features_dir}")
            return
        
        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of CSV files
        csv_files = glob(os.path.join(features_dir, "*.csv"))
        if not csv_files:
            logging.error(f"No CSV files found in directory: {features_dir}")
            return
        
        logging.info(f"Found {len(csv_files)} CSV files to process.")
        
        # First pass: Collect all feature names
        logging.info("Collecting all feature names from CSV files.")
        feature_names = set()
        for idx, csv_file in enumerate(csv_files):
            logging.info(f"Collecting feature names from file {idx+1}/{len(csv_files)}: {csv_file}")
            df = pd.read_csv(csv_file, nrows=0)
            feature_names.update(df.columns.tolist())
            del df
            gc.collect()
        
        feature_names = sorted(list(feature_names))
        logging.info(f"Total number of features: {len(feature_names)}")
        
        # First pass: Fit the scaler incrementally
        logging.info("Starting first pass to fit the scaler.")
        scaler = StandardScaler()
        
        for idx, csv_file in enumerate(csv_files):
            logging.info(f"Fitting scaler on file {idx+1}/{len(csv_files)}: {csv_file}")
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                chunk = chunk.fillna(0)
                chunk = chunk.reindex(columns=feature_names, fill_value=0)
                scaler.partial_fit(chunk)
                del chunk
                gc.collect()
        
        logging.info("Scaler fitting complete.")
        
        # Save the scaler
        joblib.dump(scaler, scaler_file)
        logging.info(f"Scaler saved to {scaler_file}")
        
        # Second pass: Transform data and save scaled features
        logging.info("Starting second pass to transform data and save scaled features.")
        for idx, csv_file in enumerate(csv_files):
            logging.info(f"Transforming data from file {idx+1}/{len(csv_files)}: {csv_file}")
            output_file = os.path.join(output_dir, os.path.basename(csv_file))
            first_chunk = True
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                chunk = chunk.fillna(0)
                chunk = chunk.reindex(columns=feature_names, fill_value=0)
                scaled_chunk = scaler.transform(chunk)
                scaled_df = pd.DataFrame(scaled_chunk, columns=feature_names)
                if first_chunk:
                    scaled_df.to_csv(output_file, index=False, mode='w')
                    first_chunk = False
                else:
                    scaled_df.to_csv(output_file, index=False, mode='a', header=False)
                del chunk
                del scaled_chunk
                del scaled_df
                gc.collect()
            logging.info(f"Scaled features saved to {output_file}")
        
        logging.info("Data transformation complete.")
        
    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("features_dir", help="Path to the directory containing feature CSV files.")
    parser.add_argument("--scaler_file", default="output/scaler.pkl", help="File to save the scaler.")
    parser.add_argument("--output_dir", default="output/scaled_features", help="Directory to save the scaled features.")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Number of rows per chunk when reading CSV files.")
    args = parser.parse_args()
    
    prepare_dataset(args.features_dir, args.scaler_file, args.output_dir, chunk_size=args.chunk_size)
