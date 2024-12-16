import os
import pickle
import json
import logging
import numpy as np

def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def convert_numpy_to_list(data):
    """
    Recursively convert NumPy arrays to lists and non-serializable objects to strings where necessary.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {str(key): convert_numpy_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_numpy_to_list(item) for item in data)
    else:
        return data

def convert_pkl_to_json(pkl_file_path, json_file_path):
    """
    Convert a pickle file to a JSON file, handling non-serializable objects.
    """
    try:
        # Load data from the pickle file
        with open(pkl_file_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        # Convert data to JSON-serializable format
        json_serializable_data = convert_numpy_to_list(data)

        # Save the data to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(json_serializable_data, json_file, indent=4)

        logging.info(f"Successfully converted {pkl_file_path} to {json_file_path}")

    except Exception as e:
        logging.error(f"Failed to convert {pkl_file_path} to {json_file_path}: {e}")

def process_directory(input_dir, output_dir, substring):
    """
    Process all files in a directory that contain a specific substring.
    Convert matching .pkl files to .json.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    for root, _, files in os.walk(input_dir):
        for file in files:
            if substring in file and file.endswith(".pkl"):
                pkl_file_path = os.path.join(root, file)
                json_file_name = file.replace(".pkl", ".json")
                json_file_path = os.path.join(output_dir, json_file_name)

                logging.info(f"Processing file: {pkl_file_path}")
                convert_pkl_to_json(pkl_file_path, json_file_path)

if __name__ == "__main__":
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description="Convert pickle files in a directory to JSON files.")
    parser.add_argument("--input-dir", required=True, help="Path to the input directory containing pickle files")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory for JSON files")
    parser.add_argument("--substring", default="anomaly_scores.pkl", help="Substring to filter files for conversion")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.substring)
