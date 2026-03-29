"""
MVTec AD dataset loader.

Folder structure expected:
    data/mvtec/<category>/train/good/       ← normal training images
    data/mvtec/<category>/test/good/        ← normal test images
    data/mvtec/<category>/test/<defect>/    ← anomalous test images
    data/mvtec/<category>/ground_truth/<defect>/  ← binary masks (optional)
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(image_size: int, augment: bool = False):
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        base = aug + base
    return transforms.Compose(base)


class MVTecDataset(Dataset):
    """
    Returns (image_tensor, label, defect_type) where:
        label = 0  → normal
        label = 1  → anomalous
    """

    def __init__(self, root: str, category: str, split: str,
                 image_size: int = 256, augment: bool = False):
        assert split in ("train", "test")
        self.split = split
        self.transform = get_transforms(image_size, augment and split == "train")

        split_dir = Path(root) / category / split
        self.samples = []   # list of (path, label, defect_type)

        for defect_dir in sorted(split_dir.iterdir()):
            label = 0 if defect_dir.name == "good" else 1
            for img_path in sorted(defect_dir.glob("*.png")) + \
                            sorted(defect_dir.glob("*.jpg")):
                self.samples.append((img_path, label, defect_dir.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, defect_type = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, defect_type

    def normal_only(self):
        """Return a copy of this dataset filtered to normal samples only."""
        ds = MVTecDataset.__new__(MVTecDataset)
        ds.split = self.split
        ds.transform = self.transform
        ds.samples = [(p, l, d) for p, l, d in self.samples if l == 0]
        return ds
