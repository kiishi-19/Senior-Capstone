import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from glob import glob
from multiprocessing import Pool

# Setup logging
logging.basicConfig(filename='output/inference_autoencoder.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], bottleneck_dim=8, activation_fn=nn.ReLU,
                 dropout_prob=0.2):
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
    def __init__(self, pkl_file, expected_features=None):
        self.data = pd.read_pickle(pkl_file)

        # Validate features
        if expected_features:
            missing_features = [f for f in expected_features if f not in self.data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

        self.data = self.data.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_model(model_path, input_dim, hidden_dims, bottleneck_dim, activation_fn):
    try:
        model = Autoencoder(input_dim, hidden_dims=hidden_dims, bottleneck_dim=bottleneck_dim, activation_fn=activation_fn)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None

def classify_anomalies(reconstruction_loss, thresholds):
    """Classify anomalies based on reconstruction loss and multiple thresholds."""
    classifications = {
        threshold: (reconstruction_loss > threshold).tolist() for threshold in thresholds
    }
    return classifications

def run_inference(models, data_loader, thresholds, device='cpu'):
    """Run inference and classify anomalies using multiple thresholds."""
    all_results = []

    for model in models:
        model.to(device)
        model.eval()

        with torch.no_grad():
            results = []
            for batch in data_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                reconstruction_loss = nn.MSELoss(reduction='none')(reconstructed, batch).mean(dim=1).cpu().numpy()

                # Classify anomalies for each threshold
                classifications = classify_anomalies(reconstruction_loss, thresholds)
                results.append((reconstruction_loss, classifications))

            all_results.append(results)
    return all_results

def process_file(file_path, models, expected_features, thresholds, batch_size, device):
    try:
        # Load dataset
        inference_dataset = InferenceDataset(file_path, expected_features)
        data_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

        # Run inference
        results = run_inference(models, data_loader, thresholds, device=device)

        # Save anomaly scores and classifications
        output_scores = os.path.splitext(file_path)[0] + "_anomaly_scores.npy"
        np.save(output_scores, results)
        logging.info(f"Anomaly scores saved to {output_scores}")

        return output_scores
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def process_files_in_parallel(file_paths, models, expected_features, thresholds, batch_size, device):
    with Pool() as pool:
        results = pool.starmap(
            process_file,
            [(file, models, expected_features, thresholds, batch_size, device) for file in file_paths]
        )
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with trained autoencoder models.")
    parser.add_argument("inference_data_dir", help="Directory containing inference data")
    parser.add_argument("--model_paths", nargs="+", required=True, help="Paths to the trained model files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 32, 16], help="Hidden layer dimensions")
    parser.add_argument("--bottleneck_dim", type=int, default=8, help="Bottleneck layer dimension")
    parser.add_argument("--activation", choices=["sigmoid", "relu"], default="relu", help="Activation function")
    parser.add_argument("--dropout_prob", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7], help="Anomaly thresholds")
    args = parser.parse_args()

    activation_fn = nn.Sigmoid if args.activation == "sigmoid" else nn.ReLU

    # Load all models
    feature_dim = pd.read_pickle(glob(os.path.join(args.inference_data_dir, '*.pkl'))[0]).shape[1]
    models = [
        load_model(model_path, input_dim=feature_dim, hidden_dims=args.hidden_dims, bottleneck_dim=args.bottleneck_dim, activation_fn=activation_fn)
        for model_path in args.model_paths
    ]
    models = [model for model in models if model]  # Remove any models that failed to load

    # Validate and prepare file paths
    file_paths = [os.path.join(args.inference_data_dir, f) for f in os.listdir(args.inference_data_dir) if f.endswith(".pkl")]

    # Run parallel inference
    process_files_in_parallel(file_paths, models, feature_dim, args.thresholds, args.batch_size, args.device)
