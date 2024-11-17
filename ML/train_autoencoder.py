# train_autoencoder.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import joblib
import argparse
from sklearn.model_selection import KFold
import numpy as np

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims=[2], bottleneck_dim=1, activation_fn=nn.Sigmoid):
        super(Autoencoder, self).__init__()
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
        x = batch[0]
        reconstructed = self.forward(x)
        loss = self.criterion(reconstructed, x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        reconstructed = self.forward(x)
        val_loss = self.criterion(reconstructed, x)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def train_autoencoder_cross_validation(scaled_features_file, model_output_file='output/autoencoder_model.pth',
                                       max_epochs=120, hidden_dims=[2], bottleneck_dim=1, activation_fn=nn.Sigmoid,
                                       n_splits=4):
    # Load scaled features
    data = joblib.load(scaled_features_file)
    scaled_features = data['scaled_features']
    feature_names = data['feature_names']

    # Prepare dataset
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    dataset = TensorDataset(features_tensor)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 0
    val_losses = []

    for train_index, val_index in kf.split(features_tensor):
        fold += 1
        print(f"\nStarting fold {fold}/{n_splits}")

        train_subset = torch.utils.data.Subset(dataset, train_index)
        val_subset = torch.utils.data.Subset(dataset, val_index)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        # Determine input dimension
        input_dim = features_tensor.shape[1]

        # Instantiate model
        model = Autoencoder(input_dim, hidden_dims=hidden_dims, bottleneck_dim=bottleneck_dim, activation_fn=activation_fn)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
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

        # Save the model for this fold
        fold_model_output_file = model_output_file.replace('.pth', f'_fold{fold}.pth')
        torch.save(model.state_dict(), fold_model_output_file)
        print(f"Model for fold {fold} saved to {fold_model_output_file}")

    # Calculate average validation loss across folds
    avg_val_loss = np.mean(val_losses)
    print(f"\nAverage validation loss across {n_splits} folds: {avg_val_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autoencoder model with cross-validation.")
    parser.add_argument("scaled_features_file", help="Path to the scaled features file.")
    parser.add_argument("--model_output_file", default="autoencoder_model.pth", help="File to save the trained model.")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs for training.")
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[2], help="List of hidden layer sizes.")
    parser.add_argument("--bottleneck_dim", type=int, default=1, help="Size of the bottleneck layer.")
    parser.add_argument("--activation", choices=["sigmoid", "relu"], default="sigmoid", help="Activation function to use.")
    parser.add_argument("--n_splits", type=int, default=4, help="Number of folds for cross-validation.")
    args = parser.parse_args()

    # Map activation function argument to class
    activation_fn = nn.Sigmoid if args.activation == "sigmoid" else nn.ReLU

    train_autoencoder_cross_validation(
        args.scaled_features_file,
        model_output_file=args.model_output_file,
        max_epochs=args.epochs,
        hidden_dims=args.hidden_dims,
        bottleneck_dim=args.bottleneck_dim,
        activation_fn=activation_fn,
        n_splits=args.n_splits
    )
