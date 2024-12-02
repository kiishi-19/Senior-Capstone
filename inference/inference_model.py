import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import logging
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from glob import glob

# Setup logging
logging.basicConfig(filename='output/inference_autoencoder.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims=[2], bottleneck_dim=1, activation_fn=nn.Sigmoid,
                 dropout_prob=0.2, learning_rate=1e-4):
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                activation_fn(),
                nn.Dropout(dropout_prob),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        encoder_layers.append(activation_fn())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                activation_fn(),
                nn.Dropout(dropout_prob),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(activation_fn())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class InferenceDataset(Dataset):
    def __init__(self, pkl_file):
        self.data = pd.read_pickle(pkl_file).values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_model(model_path, input_dim, hidden_dims, bottleneck_dim, activation_fn):
    model = Autoencoder(input_dim, hidden_dims=hidden_dims, bottleneck_dim=bottleneck_dim, activation_fn=activation_fn)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def run_inference(models, data_loader, baseline_threshold=7.5, device='cpu'):
    """Run inference and flag anomalies based on the baseline MSE threshold."""
    all_scores = []
    for model in models:
        model.to(device)
        model.eval()
        scores = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                mse_loss = nn.MSELoss(reduction='none')(reconstructed, batch).mean(dim=1)

                # Flag anomalies based on the baseline threshold
                anomalies = mse_loss > baseline_threshold
                scores.extend(zip(mse_loss.cpu().numpy(), anomalies.cpu().numpy()))

        all_scores.append(scores)
    return all_scores

def process_anomaly_scores(scores, threshold, file_name):
    # Convert scores to numpy array if it's not already
    scores = np.array(scores)
    
    logging.info(f"\nAnalyzing scores for {file_name}:")
    logging.info(f"Score statistics:")
    logging.info(f"- Min score: {np.min(scores):.4f}")
    logging.info(f"- Max score: {np.max(scores):.4f}")
    logging.info(f"- Mean score: {np.mean(scores):.4f}")
    logging.info(f"- Median score: {np.median(scores):.4f}")
    logging.info(f"- Std dev: {np.std(scores):.4f}")
    
    anomalies = scores > threshold
    anomaly_percentage = (anomalies.sum() / len(scores)) * 100
    logging.info(f"Threshold: {threshold:.4f}")
    logging.info(f"Detected anomalies: {anomaly_percentage:.2f}%")
    
    if anomaly_percentage > 0:
        logging.info(f"Anomaly scores above threshold: {scores[anomalies]}")
    
    return anomaly_percentage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained autoencoder model.")
    parser.add_argument("inference_data_dir", help="Directory containing inference data")
    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
        help="Paths to the trained model files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[64, 32, 16],
        help="Hidden layer dimensions"
    )
    parser.add_argument(
        "--bottleneck_dim",
        type=int,
        default=8,
        help="Bottleneck layer dimension"
    )
    parser.add_argument(
        "--activation",
        choices=["sigmoid", "relu"],
        default="relu",
        help="Activation function"
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.2,
        help="Dropout probability"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    
    args = parser.parse_args()

    activation_fn = nn.Sigmoid if args.activation == "sigmoid" else nn.ReLU

    # Load each saved model
    feature_dim = pd.read_pickle(glob(os.path.join(args.inference_data_dir, '*.pkl'))[0]).shape[1]  # Get input dim from data
    models = [
        load_model(model_path, input_dim=feature_dim, hidden_dims=args.hidden_dims, bottleneck_dim=args.bottleneck_dim, activation_fn=activation_fn)
        for model_path in args.model_paths
    ]

    # Iterate through each .pkl file in the inference data directory
    for pkl_file in os.listdir(args.inference_data_dir):
        if pkl_file.endswith(".pkl"):
            file_path = os.path.join(args.inference_data_dir, pkl_file)
            print(f"Running inference on {file_path}")
            logging.info(f"Running inference on {file_path}")

            # Load data and create DataLoader
            inference_dataset = InferenceDataset(file_path)
            data_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)

            # Run inference and collect scores
            all_scores = run_inference(
                models,
                data_loader,
                baseline_threshold=7.5,  # Use the baseline threshold for anomaly detection
                device=args.device
            )

            # Save scores for each model and log results
            for model_idx, (model_path, scores) in enumerate(zip(args.model_paths, all_scores), 1):
                scores_array = np.array(scores)  # Convert to numpy array
                output_file = os.path.join(args.inference_data_dir, f"{os.path.basename(pkl_file)}_fold{model_idx}_scores.npy")
                np.save(output_file, scores_array)
                logging.info(f"Anomaly scores for {os.path.basename(pkl_file)} model fold {model_idx} saved to {output_file}")
                process_anomaly_scores(scores_array, 7.5, pkl_file)
