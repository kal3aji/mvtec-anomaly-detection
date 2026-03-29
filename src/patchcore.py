"""
PatchCore anomaly detection (Roth et al., 2022).

Key idea:
    1. Extract patch-level features from a pretrained CNN (no training needed)
    2. Build a memory bank from normal images
    3. At inference, find the nearest memory patch — high distance = anomaly

Reference: https://arxiv.org/abs/2106.08265
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class PatchCore:
    def __init__(self, backbone: str = "wide_resnet50_2",
                 layers: list = None, coreset_n: int = 1000,
                 device: str = "cpu"):
        self.device = device
        self.layers = layers or ["layer2", "layer3"]
        self.coreset_n = coreset_n
        self._features = {}
        self._hooks = []

        # Load pretrained backbone (frozen)
        backbone_fn = getattr(models, backbone)
        self.model = backbone_fn(weights="IMAGENET1K_V1").to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Register forward hooks on requested layers
        for name, module in self.model.named_modules():
            if name in self.layers:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

        self.memory_bank: np.ndarray | None = None
        self.nn_index: NearestNeighbors | None = None

    def _make_hook(self, name: str):
        def hook(_, __, output):
            self._features[name] = output
        return hook

    def _extract_patches(self, x: torch.Tensor) -> np.ndarray:
        """
        Forward pass → collect feature maps → reshape into patch descriptors.
        Returns array of shape (N_patches, feature_dim).
        """
        self._features.clear()
        with torch.no_grad():
            self.model(x.to(self.device))

        patch_list = []
        for name in self.layers:
            feat = self._features[name]               # (B, C, H, W)
            # Upsample all layers to the same spatial size (first layer size)
            target_h = self._features[self.layers[0]].shape[2]
            target_w = self._features[self.layers[0]].shape[3]
            feat = F.interpolate(feat, size=(target_h, target_w),
                                 mode="bilinear", align_corners=False)
            patch_list.append(feat)

        # Concatenate along channel dim → (B, C_total, H, W)
        patches = torch.cat(patch_list, dim=1)
        B, C, H, W = patches.shape
        # → (B*H*W, C)
        return patches.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, dataloader) -> None:
        """Build memory bank from normal training images."""
        all_patches = []
        print("Building PatchCore memory bank...")
        for imgs, _, _ in tqdm(dataloader):
            all_patches.append(self._extract_patches(imgs))

        bank = np.vstack(all_patches)   # (N_total, C)

        # Greedy coreset subsampling — keeps a diverse, compact memory bank
        if len(bank) > self.coreset_n:
            idx = self._coreset_sample(bank, self.coreset_n)
            bank = bank[idx]

        self.memory_bank = bank
        self.nn_index = NearestNeighbors(n_neighbors=1, metric="euclidean",
                                         algorithm="ball_tree", n_jobs=-1)
        self.nn_index.fit(bank)
        print(f"Memory bank built: {len(bank)} patches, dim={bank.shape[1]}")

    def _coreset_sample(self, features: np.ndarray, n: int) -> np.ndarray:
        """Random subsample (fast). Replace with greedy coreset for best results."""
        return np.random.choice(len(features), size=n, replace=False)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, x: torch.Tensor):
        """
        Returns:
            image_score  : float — max nearest-neighbour distance (image-level)
            anomaly_map  : np.ndarray (H, W) — per-patch score map
        """
        assert self.nn_index is not None, "Call fit() before predict()"
        patches = self._extract_patches(x)                  # (N, C)
        dists, _ = self.nn_index.kneighbors(patches)        # (N, 1)
        dists = dists.flatten()

        # Reshape to spatial map
        h = w = int(np.sqrt(len(dists)))
        anomaly_map = dists[:h * w].reshape(h, w)

        image_score = float(dists.max())
        return image_score, anomaly_map

    def cleanup(self):
        """Remove forward hooks."""
        for h in self._hooks:
            h.remove()
