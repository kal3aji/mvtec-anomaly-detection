"""
Build the PatchCore memory bank (no gradient training — just one forward pass).

Usage:
    python src/train_patchcore.py
"""

import os
import sys
import pickle
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.dataset import MVTecDataset
from src.patchcore import PatchCore
from configs.config import (
    CATEGORY, DATA_ROOT, IMAGE_SIZE, BATCH_SIZE,
    PATCHCORE_BACKBONE, PATCHCORE_LAYERS, PATCHCORE_CORESET_N, MODEL_DIR
)


def build_memory_bank():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Category: {CATEGORY}")
    print(f"Backbone: {PATCHCORE_BACKBONE}  |  Layers: {PATCHCORE_LAYERS}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MVTecDataset(DATA_ROOT, CATEGORY, "train",
                            IMAGE_SIZE, augment=False).normal_only()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)
    print(f"Normal training images: {len(train_ds)}")

    # ── PatchCore ─────────────────────────────────────────────────────────────
    pc = PatchCore(backbone=PATCHCORE_BACKBONE, layers=PATCHCORE_LAYERS,
                   coreset_n=PATCHCORE_CORESET_N, device=device)
    pc.fit(train_loader)

    # Save memory bank
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = f"{MODEL_DIR}/patchcore_{CATEGORY}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"memory_bank": pc.memory_bank}, f)

    print(f"Memory bank saved to {save_path}")
    pc.cleanup()


if __name__ == "__main__":
    build_memory_bank()
