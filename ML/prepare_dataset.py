

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse

def prepare_dataset(features_file, scaler_file='output/scaler.pkl', output_file='output/scaled_features.pkl'):
    """
    Load features from a file, scale them, and save the scaler and scaled features.

    Args:
        features_file (str): Path to the features file (CSV or Parquet).
        scaler_file (str): Path to save the scaler.
        output_file (str): Path to save the scaled features.
    """
    # Load features
    if features_file.endswith('.csv'):
        features_df = pd.read_csv(features_file)
    elif features_file.endswith('.parquet'):
        features_df = pd.read_parquet(features_file)
    else:
        print(f"Unsupported features file format: {features_file}")
        return

    # Handle missing values again incase
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("features_file", help="Path to the features file (CSV or Parquet).")
    parser.add_argument("--scaler_file", default="scaler.pkl", help="File to save the scaler.")
    parser.add_argument("--output_file", default="scaled_features.pkl", help="File to save the scaled features.")
    args = parser.parse_args()

    prepare_dataset(args.features_file, args.scaler_file, args.output_file)
