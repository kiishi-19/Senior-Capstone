# train_autoencoder.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import joblib
import argparse

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
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def train_autoencoder(scaled_features_file, model_output_file='output/autoencoder_model.pth', max_epochs=120, hidden_dims=[2], bottleneck_dim=1, activation_fn=nn.Sigmoid):
    # Load scaled features
    data = joblib.load(scaled_features_file)
    scaled_features = data['scaled_features']
    feature_names = data['feature_names']

    # Prepare DataLoader
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

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
                dirpath='checkpoints/',
                filename='best_model',
                save_top_k=1,
                monitor='train_loss',
                mode='min'
            )
        ]
    )

    # Train model
    trainer.fit(model, dataloader)

    # Save the trained model
    torch.save(model.state_dict(), model_output_file)
    print(f"Model saved to {model_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autoencoder model.")
    parser.add_argument("scaled_features_file", help="Path to the scaled features file.")
    parser.add_argument("--model_output_file", default="autoencoder_model.pth", help="File to save the trained model.")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs for training.")
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[2], help="List of hidden layer sizes.")
    parser.add_argument("--bottleneck_dim", type=int, default=1, help="Size of the bottleneck layer.")
    parser.add_argument("--activation", choices=["sigmoid", "relu"], default="sigmoid", help="Activation function to use.")
    args = parser.parse_args()

    # Map activation function argument to class
    activation_fn = nn.Sigmoid if args.activation == "sigmoid" else nn.ReLU

    train_autoencoder(
        args.scaled_features_file,
        model_output_file=args.model_output_file,
        max_epochs=args.epochs,
        hidden_dims=args.hidden_dims,
        bottleneck_dim=args.bottleneck_dim,
        activation_fn=activation_fn
    )
