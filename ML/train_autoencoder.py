import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import logging
import gc

print(torch.cuda.is_available())  # Should return True if a GPU is available

# Configure logging
logging.basicConfig(filename='output/train_autoencoder.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims=[2], bottleneck_dim=1, activation_fn=nn.Sigmoid):
        super(Autoencoder, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                activation_fn(),
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
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(activation_fn())
        self.decoder = nn.Sequential(*decoder_layers)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x = batch
        reconstructed = self.forward(x)
        loss = self.criterion(reconstructed, x)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        logging.info(f"Training loss at batch {batch_idx}: {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        reconstructed = self.forward(x)
        val_loss = self.criterion(reconstructed, x)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        logging.info(f"Validation loss at batch {batch_idx}: {val_loss.item()}")
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ScaledFeaturesDataset(Dataset):
    def __init__(self, pkl_file, feature_names):
        self.data = pd.read_pickle(pkl_file)
        self.feature_names = feature_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx][self.feature_names].values.astype(np.float32)

def train_autoencoder_cross_validation(pkl_file, model_output_file='output/autoencoder_model.pth',
                                       max_epochs=120, hidden_dims=[2], bottleneck_dim=1, activation_fn=nn.Sigmoid,
                                       n_splits=4, batch_size=64):
    # Load feature names from the pickle file
    try:
        feature_names = pd.read_pickle(pkl_file).columns.tolist()
    except Exception as e:
        logging.error(f"Error reading feature names from pickle file {pkl_file}: {e}")
        return

    # Create the dataset
    dataset = ScaledFeaturesDataset(pkl_file, feature_names)
    input_dim = len(feature_names)
    logging.info(f"Input dimension (number of features): {input_dim}")

    # Prepare indices for K-Fold Cross-Validation
    indices = list(range(len(dataset)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 0
    val_losses = []

    for train_indices, val_indices in kf.split(indices):
        fold += 1
        print(f"\nStarting fold {fold}/{n_splits}")
        logging.info(f"Starting fold {fold}/{n_splits}")

        # Create data loaders for training and validation
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4,
                                  collate_fn=lambda x: torch.stack([torch.tensor(item) for item in x]))
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4,
                                collate_fn=lambda x: torch.stack([torch.tensor(item) for item in x]))

        # Instantiate model
        model = Autoencoder(input_dim, hidden_dims=hidden_dims, bottleneck_dim=bottleneck_dim, activation_fn=activation_fn)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=1 if torch.cuda.is_available() else None,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=f'checkpoints/fold_{fold}/',
                    filename='best_model',
                    save_top_k=1,
                    monitor='val_loss',
                    mode='min'
                )
            ],
            logger=False  # Disable logging if not needed
        )

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Evaluate on validation set
        val_result = trainer.validate(model, val_loader)
        val_loss = val_result[0]['val_loss']
        val_losses.append(val_loss)
        logging.info(f"Validation loss for fold {fold}: {val_loss}")

        # Save the model for this fold
        fold_model_output_file = model_output_file.replace('.pth', f'_fold{fold}.pth')
        torch.save(model.state_dict(), fold_model_output_file)
        print(f"Model for fold {fold} saved to {fold_model_output_file}")
        logging.info(f"Model for fold {fold} saved to {fold_model_output_file}")

        # Clean up to free memory
        del model
        gc.collect()

    # Calculate average validation loss across folds
    avg_val_loss = np.mean(val_losses)
    print(f"\nAverage validation loss across {n_splits} folds: {avg_val_loss:.6f}")
    logging.info(f"Average validation loss across {n_splits} folds: {avg_val_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autoencoder model with cross-validation.")
    parser.add_argument("pkl_file", help="Path to the pickle file containing scaled feature data.")
    parser.add_argument("--model_output_file", default="autoencoder_model.pth", help="File to save the trained model.")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs for training.")
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[2], help="List of hidden layer sizes.")
    parser.add_argument("--bottleneck_dim", type=int, default=1, help="Size of the bottleneck layer.")
    parser.add_argument("--activation", choices=["sigmoid", "relu"], default="sigmoid", help="Activation function to use.")
    parser.add_argument("--n_splits", type=int, default=4, help="Number of folds for cross-validation.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    args = parser.parse_args()

    # Map activation function argument to class
    activation_fn = nn.Sigmoid if args.activation == "sigmoid" else nn.ReLU

    train_autoencoder_cross_validation(
        args.pkl_file,
        model_output_file=args.model_output_file,
        max_epochs=args.epochs,
        hidden_dims=args.hidden_dims,
        bottleneck_dim=args.bottleneck_dim,
        activation_fn=activation_fn,
        n_splits=args.n_splits,
        batch_size=args.batch_size
    )
