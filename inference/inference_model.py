# inference_autoencoder_with_confidence.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import logging
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(filename='output/inference_autoencoder.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims=[2], bottleneck_dim=1, activation_fn=nn.Sigmoid):
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), activation_fn()])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        encoder_layers.append(activation_fn())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), activation_fn()])
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

def calculate_anomaly_confidence(mse_scores, threshold=None):
    """Convert MSE scores to a 0-100 confidence score scale based on the threshold."""
    max_score = mse_scores.max()
    confidence_scores = (mse_scores / max_score) * 100
    return confidence_scores

def run_inference(models, data_loader, confidence_threshold=90, thresholding_enabled=False, device='cpu'):
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
                confidence_scores = calculate_anomaly_confidence(mse_loss)  # Convert MSE to confidence

                if thresholding_enabled:
                    anomalies = confidence_scores > confidence_threshold  # Flag if confidence is above threshold
                    scores.extend(zip(mse_loss.cpu().numpy(), confidence_scores.cpu().numpy(), anomalies.cpu().numpy()))
                else:
                    scores.extend(zip(mse_loss.cpu().numpy(), confidence_scores.cpu().numpy()))

        all_scores.append(scores)
    return all_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with confidence thresholding on saved autoencoder models.")
    parser.add_argument("inference_data_dir", help="Path to the directory containing scaled inference .pkl files.")
    parser.add_argument("--model_paths", nargs='+', required=True, help="List of paths to saved model files for each fold.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[2], help="List of hidden layer sizes used in model.")
    parser.add_argument("--bottleneck_dim", type=int, default=1, help="Size of the bottleneck layer used in model.")
    parser.add_argument("--activation", choices=["sigmoid", "relu"], default="sigmoid", help="Activation function to use.")
    parser.add_argument("--confidence_threshold", type=float, default=90.0, help="Confidence threshold for anomaly detection.")
    parser.add_argument("--thresholding_enabled", action='store_true', help="Enable thresholding for anomaly detection.")
    parser.add_argument("--device", default="cpu", help="Device to run inference on (e.g., 'cpu' or 'cuda').")
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
                confidence_threshold=args.confidence_threshold, 
                thresholding_enabled=args.thresholding_enabled, 
                device=args.device
            )

            # Save scores for each model and log results
            for i, scores in enumerate(all_scores):
                output_path = os.path.join(args.inference_data_dir, f"{pkl_file}_fold{i+1}_scores.npy")
                np.save(output_path, scores)
                print(f"Anomaly scores for {pkl_file} model fold {i+1} saved to {output_path}")
                logging.info(f"Anomaly scores for {pkl_file} model fold {i+1} saved to {output_path}")

                if args.thresholding_enabled:
                    num_anomalous = sum(1 for _, _, is_anomalous in scores if is_anomalous)
                    anomaly_percentage = (num_anomalous / len(scores)) * 100
                    print(f"Thresholding enabled. Anomaly confidence threshold: {args.confidence_threshold}%. Detected anomalies: {anomaly_percentage:.2f}%")
                    logging.info(f"Thresholding enabled. Anomaly confidence threshold: {args.confidence_threshold}%. Detected anomalies: {anomaly_percentage:.2f}%")
