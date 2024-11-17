import os
import pandas as pd
import joblib
from glob import glob
import argparse
from sklearn.preprocessing import StandardScaler

def prepare_dataset(features_dir, scaler_file='output/scaler.pkl', output_file='output/scaled_features.pkl'):
    """
    Load features from Parquet files in a directory, scale them, and save the scaler and scaled features.
    """
    try:
        # Read all Parquet files in the directory
        parquet_files = glob(os.path.join(features_dir, "*.parquet"))
        if not parquet_files:
            print(f"No Parquet files found in directory: {features_dir}")
            return

        # Read and concatenate all Parquet files
        features_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        # Handle missing values
        features_df = features_df.fillna(0)

        # Separate feature names for later use
        feature_names = features_df.columns.tolist()

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)

        # Save the scaler
        joblib.dump(scaler, scaler_file)
        print(f"Scaler saved to {scaler_file}")

        # Save the scaled features and feature names
        joblib.dump({
            'scaled_features': scaled_features,
            'feature_names': feature_names
        }, output_file)
        print(f"Scaled features saved to {output_file}")

        return scaled_features, feature_names

    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("features_dir", help="Path to the directory containing feature Parquet files.")
    parser.add_argument("--scaler_file", default="output/scaler.pkl", help="File to save the scaler.")
    parser.add_argument("--output_file", default="output/scaled_features.pkl", help="File to save the scaled features.")
    args = parser.parse_args()

    prepare_dataset(args.features_dir, args.scaler_file, args.output_file)
