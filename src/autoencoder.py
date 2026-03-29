"""
Convolutional Autoencoder for anomaly detection.

Training strategy:
    - Train ONLY on normal (good) images
    - At inference, high reconstruction error → anomaly

Architecture: encoder compresses to a latent vector,
              decoder reconstructs the original image.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Small residual block to improve reconstruction quality."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # 256 → 128
            nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # 128 → 64
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64),
            # 64 → 32
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResidualBlock(128),
            # 32 → 16
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, x):
        feat = self.net(x)
        return self.fc(feat.view(feat.size(0), -1))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.net = nn.Sequential(
            # 16 → 32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResidualBlock(128),
            # 32 → 64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64),
            # 64 → 128
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # 128 → 256
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 16, 16)
        return self.net(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-image anomaly score = mean squared reconstruction error.
        Shape: (B,) — higher means more anomalous.
        """
        with torch.no_grad():
            recon = self.forward(x)
            # Per-pixel MSE, then average over C,H,W
            score = ((x - recon) ** 2).mean(dim=(1, 2, 3))
        return score

    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-pixel anomaly map for heatmap visualization.
        Shape: (B, H, W)
        """
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=1)
