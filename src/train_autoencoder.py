"""
Train the Convolutional Autoencoder on normal (good) images only.

Usage:
    python src/train_autoencoder.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.dataset import MVTecDataset
from src.autoencoder import ConvAutoencoder
from configs.config import (
    CATEGORY, DATA_ROOT, IMAGE_SIZE,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, LATENT_DIM, MODEL_DIR
)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}  |  Category: {CATEGORY}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MVTecDataset(DATA_ROOT, CATEGORY, "train",
                            IMAGE_SIZE, augment=True).normal_only()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    print(f"Training samples: {len(train_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for imgs, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}",
                                leave=False):
            imgs = imgs.to(device)
            recon = model(imgs)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       f"{MODEL_DIR}/autoencoder_{CATEGORY}_best.pth")

    print(f"\nDone. Best loss: {best_loss:.6f}")
    print(f"Model saved to {MODEL_DIR}/autoencoder_{CATEGORY}_best.pth")


if __name__ == "__main__":
    train()
