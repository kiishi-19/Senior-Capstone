from pathlib import Path
import streamlit as st
import pandas as pd
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from inference_process_scapfiles import process_scap_file_inference
from inference_prepare_dataset import prepare_inference_data
from inference_model import load_model, run_inference, InferenceDataset
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler()])

st.title("SCAP Inference Dashboard")
st.markdown("""
Upload `.scap` files, process them, scale features, and run anomaly detection using pre-trained models.
Required known syscalls, arguments, and scaler files are loaded relative to the project directory.
""")

# Relative paths
KNOWN_SYSCALLS_PATH = project_root / "output/features/known_syscalls.pkl"
KNOWN_ARGUMENTS_PATH = project_root / "output/features/known_arguments.pkl"
SCALER_PATH = project_root / "output/scaler.pkl"
MODEL_PATHS = [
    project_root / "output/models/autoencoder_model_fold1.pth",
    project_root / "output/models/autoencoder_model_fold2.pth",
    project_root / "output/models/autoencoder_model_fold3.pth",
    project_root / "output/models/autoencoder_model_fold4.pth",
]
OUTPUT_DIR = project_root / "output/inference_results"

# Create necessary directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File Upload Section: Only SCAP files
st.header("Upload SCAP Files")
scap_files = st.file_uploader("Upload SCAP Files", type=["scap"], accept_multiple_files=True)

# Process and Run Inference Button
if st.button("Run"):
    if not scap_files:
        st.error("Please upload at least one SCAP file.")
    else:
        try:
            # Save uploaded SCAP files
            st.write("Processing uploaded SCAP files...")
            scap_paths = []
            for scap_file in scap_files:
                scap_path = OUTPUT_DIR / scap_file.name
                with open(scap_path, "wb") as f:
                    f.write(scap_file.getbuffer())
                scap_paths.append(scap_path)

            # Process SCAP files
            for scap_path in scap_paths:
                st.write(f"Processing {scap_path.name}...")
                process_scap_file_inference(
                    scap_file=scap_path,
                    output_dir=OUTPUT_DIR,
                )
            st.success("SCAP processing completed.")

            # Scale features
            st.write("Scaling features...")
            prepare_inference_data(
                features_dir=OUTPUT_DIR,
                scaler_file=SCALER_PATH,
                output_dir=OUTPUT_DIR,
                chunk_size=5000,
                n_jobs=-1,
            )
            st.success("Feature scaling completed.")

            # Run anomaly detection
            st.write("Running anomaly detection...")
            feature_dim = pd.read_pickle(list(OUTPUT_DIR.glob("*.pkl"))[0]).shape[1]
            models = [
                load_model(
                    model_path=model_path,
                    input_dim=feature_dim,
                    hidden_dims=[64, 32, 16],
                    bottleneck_dim=8,
                    activation_fn=nn.ReLU,
                )
                for model_path in MODEL_PATHS
            ]

            for pkl_file in OUTPUT_DIR.glob("*.pkl"):
                dataset = InferenceDataset(pkl_file)
                loader = DataLoader(dataset, batch_size=64, shuffle=False)
                all_scores = run_inference(models, loader, baseline_threshold=81.71176, device="cuda")

                for idx, scores in enumerate(all_scores, 1):
                    output_file = OUTPUT_DIR / f"{pkl_file.stem}_fold{idx}_scores.npy"
                    np.save(output_file, scores)
                    st.write(f"Saved anomaly scores for {pkl_file.name}, Model Fold {idx}: {output_file}")
                    st.dataframe(scores[:10])  # Display the first 10 scores

        except Exception as e:
            st.error(f"Error during processing: {e}")
