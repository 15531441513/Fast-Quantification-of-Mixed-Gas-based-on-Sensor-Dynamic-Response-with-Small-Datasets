import torch
import numpy as np
from torch import nn
import pickle
import os

class DataGenerator:
    # 获取本脚本所在目录，确保路径在任何工作目录下都正确
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATHS = [
        os.path.join(_BASE_DIR, "vae_model_0.pth"),
        os.path.join(_BASE_DIR, "vae_model_1.pth"),
        os.path.join(_BASE_DIR, "vae_model_2.pth")
    ]
    SCALER_PATHS = [
        os.path.join(_BASE_DIR, "scaler_0.pkl"),
        os.path.join(_BASE_DIR, "scaler_1.pkl"),
        os.path.join(_BASE_DIR, "scaler_2.pkl")
    ]

    class FeatureVAE(nn.Module):
        def __init__(self,latent_dim=3):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(1, 16),
                nn.LayerNorm(16),  
                nn.LeakyReLU(0.2),
                nn.Linear(16, 8),
                nn.LeakyReLU(0.2)
            )


            self.fc_mu = nn.Linear(8, latent_dim)
            self.fc_var = nn.Linear(8, latent_dim)


            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 8),
                nn.LeakyReLU(0.2),
                nn.Linear(8, 16),
                nn.LayerNorm(16),
                nn.LeakyReLU(0.2),
                nn.Linear(16, 1),
                nn.Tanh()
            )

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def forward(self, x):

            h = self.encoder(x)
            mu, logvar = self.fc_mu(h), self.fc_var(h)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z)

    @staticmethod
    def generate(num_samples=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generators = []
        scalers = []

        # Load models and scalers
        for model_path, scaler_path in zip(DataGenerator.MODEL_PATHS, DataGenerator.SCALER_PATHS):
            # Load model
            model = DataGenerator.FeatureVAE().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            generators.append(model)

            # Load scaler
            with open(scaler_path, 'rb') as f:
                scalers.append(pickle.load(f))

        # Generate data for each feature
        generated_features = []
        for model, scaler in zip(generators, scalers):
            with torch.no_grad():
                # Sample latent variables from standard normal distribution
                z = torch.randn(num_samples, 3).to(device)  # Latent dimension consistent with training

                # Generate data through decoder
                generated = model.decoder(z).cpu().numpy()

                # Inverse transform (denormalize)
                denormalized = scaler.inverse_transform(generated)
                generated_features.append(denormalized)

        # Concatenate all feature dimensions
        final_data = np.hstack(generated_features)
        return final_data

if __name__ == "__main__":
    # Test generation
    samples = DataGenerator.generate(1)
    print("Generated Data (5x4):\n", samples)
    print("Shape:", samples)