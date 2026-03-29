"""
Evaluate both models on the MVTec test set.
Computes AUROC and saves heatmap visualizations.

Usage:
    python src/evaluate.py --method autoencoder
    python src/evaluate.py --method patchcore
    python src/evaluate.py --method both
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.dataset import MVTecDataset
from src.autoencoder import ConvAutoencoder
from src.patchcore import PatchCore
from configs.config import (
    CATEGORY, DATA_ROOT, IMAGE_SIZE, BATCH_SIZE,
    LATENT_DIM, PATCHCORE_BACKBONE, PATCHCORE_LAYERS, PATCHCORE_CORESET_N,
    MODEL_DIR, RESULTS_DIR, HEATMAP_ALPHA
)


def load_autoencoder(device):
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(
        torch.load(f"{MODEL_DIR}/autoencoder_{CATEGORY}_best.pth",
                   map_location=device))
    model.eval()
    return model


def load_patchcore(device):
    save_path = f"{MODEL_DIR}/patchcore_{CATEGORY}.pkl"
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    pc = PatchCore(backbone=PATCHCORE_BACKBONE, layers=PATCHCORE_LAYERS,
                   coreset_n=PATCHCORE_CORESET_N, device=device)
    pc.memory_bank = data["memory_bank"]
    from sklearn.neighbors import NearestNeighbors
    pc.nn_index = NearestNeighbors(n_neighbors=1, metric="euclidean",
                                    algorithm="ball_tree", n_jobs=-1)
    pc.nn_index.fit(pc.memory_bank)
    return pc


def make_heatmap(original_img: np.ndarray, anomaly_map: np.ndarray,
                 save_path: str, score: float, label: int):
    """Overlay anomaly heatmap on original image and save."""
    h, w = original_img.shape[:2]
    amap_resized = cv2.resize(anomaly_map.astype(np.float32), (w, h))
    amap_norm = (amap_resized - amap_resized.min()) / \
                (amap_resized.max() - amap_resized.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * amap_norm), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 1 - HEATMAP_ALPHA,
                               heatmap, HEATMAP_ALPHA, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[1].imshow(amap_norm, cmap="hot")
    axes[1].set_title(f"Anomaly Map (score={score:.4f})")
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Overlay | GT: {'anomaly' if label else 'normal'}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def evaluate_autoencoder(device):
    print("\n── Autoencoder Evaluation ──")
    model = load_autoencoder(device)
    test_ds = MVTecDataset(DATA_ROOT, CATEGORY, "test", IMAGE_SIZE)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    all_scores, all_labels = [], []
    os.makedirs(f"{RESULTS_DIR}/heatmaps/autoencoder", exist_ok=True)

    for i, (imgs, labels, defect_types) in enumerate(tqdm(loader)):
        imgs = imgs.to(device)
        scores = model.anomaly_score(imgs)
        amaps = model.anomaly_map(imgs)

        score = scores[0].item()
        all_scores.append(score)
        all_labels.append(labels[0].item())

        # Save heatmap for first 5 anomalous and 2 normal samples
        if (labels[0] == 1 and sum(all_labels) <= 5) or \
           (labels[0] == 0 and all_labels.count(0) <= 2):
            img_np = (imgs[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            amap = amaps[0].cpu().numpy()
            save_path = f"{RESULTS_DIR}/heatmaps/autoencoder/{defect_types[0]}_{i}.png"
            make_heatmap(img_bgr, amap, save_path, score, labels[0].item())

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"  AUROC: {auroc:.4f}")
    return auroc, all_labels, all_scores


def evaluate_patchcore(device):
    print("\n── PatchCore Evaluation ──")
    pc = load_patchcore(device)
    test_ds = MVTecDataset(DATA_ROOT, CATEGORY, "test", IMAGE_SIZE)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    all_scores, all_labels = [], []
    os.makedirs(f"{RESULTS_DIR}/heatmaps/patchcore", exist_ok=True)

    for i, (imgs, labels, defect_types) in enumerate(tqdm(loader)):
        score, amap = pc.predict(imgs)
        all_scores.append(score)
        all_labels.append(labels[0].item())

        if (labels[0] == 1 and sum(all_labels) <= 5) or \
           (labels[0] == 0 and all_labels.count(0) <= 2):
            img_np = (imgs[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            save_path = f"{RESULTS_DIR}/heatmaps/patchcore/{defect_types[0]}_{i}.png"
            make_heatmap(img_bgr, amap, save_path, score, labels[0].item())

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"  AUROC: {auroc:.4f}")
    pc.cleanup()
    return auroc, all_labels, all_scores


def plot_roc_curves(results: dict):
    """Plot and save ROC curves for all evaluated methods."""
    plt.figure(figsize=(7, 6))
    for name, (labels, scores, auroc) in results.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — MVTec AD / {CATEGORY}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{RESULTS_DIR}/roc_curves_{CATEGORY}.png", dpi=150)
    print(f"\nROC curve saved to {RESULTS_DIR}/roc_curves_{CATEGORY}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["autoencoder", "patchcore", "both"],
                        default="both")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    if args.method in ("autoencoder", "both"):
        auroc, labels, scores = evaluate_autoencoder(device)
        results["Autoencoder"] = (labels, scores, auroc)

    if args.method in ("patchcore", "both"):
        auroc, labels, scores = evaluate_patchcore(device)
        results["PatchCore"] = (labels, scores, auroc)

    if len(results) > 1:
        plot_roc_curves(results)

    print("\n── Summary ──────────────────────────────────")
    print(f"Category: {CATEGORY}")
    for name, (_, _, auroc) in results.items():
        print(f"  {name:15s} AUROC: {auroc:.4f}")
