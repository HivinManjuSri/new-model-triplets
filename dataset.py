"""
dataset.py
----------
PyTorch Dataset that reads pre-built triplets from CSV files.

Each CSV row contains:
    anchor_path, positive_path, negative_path

Paths in the CSVs are absolute (from the generation machine) and are
automatically mapped to the local ``dataset/{split}/{identity}/{img}.jpg``
layout.

Returns (anchor_tensor, positive_tensor, negative_tensor) ready for the
model.  Images are loaded as JPG → RGB PIL → resized to INPUT_SIZE →
normalized to ImageNet stats.
"""

import os
import csv
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── Constants ───────────────────────────────────────────────────────────────
INPUT_SIZE   = 112
BASE_DIR     = os.path.dirname(__file__)
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
TRIPLET_DIR  = os.path.join(BASE_DIR, "triplet_dataset")

# ImageNet mean/std — used because ResNet backbone is pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# ────────────────────────────────────────────────────────────────────────────


def _make_transform(augment: bool) -> transforms.Compose:
    """
    Build a torchvision transform pipeline.

    Training augmentations are light/appropriate for face data:
      - Random horizontal flip   (faces are symmetric)
      - Small colour jitter      (lighting variations are already present)
      - NO large crops/rotations (would destroy key facial features)

    Validation / test just resizes and normalises.
    """
    steps = []
    steps.append(transforms.Resize((INPUT_SIZE, INPUT_SIZE)))

    if augment:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.02),   # rare, keeps model robust
        ]

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def _csv_path_to_local(csv_path: str) -> str:
    """
    Convert an absolute CSV path to a local dataset path.

    CSV :  F:\\DATASET\\VGGFace2\\dataset\\train\\n000208\\0194_01.jpg
    Local: dataset/train/n000208/0194_01.jpg
    """
    parts = csv_path.replace("\\", "/").split("/")
    # Find the split token (train / val / test) and keep everything from there
    for i, part in enumerate(parts):
        if part in ("train", "val", "test"):
            rel = "/".join(parts[i:])
            return os.path.join(DATASET_DIR, rel)
    # Fallback: last 3 components  (split / identity / filename)
    rel = "/".join(parts[-3:])
    return os.path.join(DATASET_DIR, rel)


class TripletFaceDataset(Dataset):
    """
    Args:
        split   : 'train', 'val', or 'test'
        augment : whether to apply training augmentations (auto True for train)
    """

    def __init__(self, split: str = "train", augment: bool | None = None):
        super().__init__()

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        if augment is None:
            augment = (split == "train")

        csv_file = os.path.join(TRIPLET_DIR, f"{split}_triplets.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"{csv_file} not found. Place the triplet CSVs in "
                f"triplet_dataset/ first."
            )

        self.triplets = []
        with open(csv_file, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            for row in reader:
                if len(row) < 3:
                    continue
                a_path = _csv_path_to_local(row[0])
                p_path = _csv_path_to_local(row[1])
                n_path = _csv_path_to_local(row[2])
                self.triplets.append((a_path, p_path, n_path))

        self.transform = _make_transform(augment)
        self.split = split
        print(f"[TripletFaceDataset] {split}: {len(self.triplets)} triplets loaded.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int):
        a_path, p_path, n_path = self.triplets[idx]

        anchor   = self.transform(Image.open(a_path).convert("RGB"))
        positive = self.transform(Image.open(p_path).convert("RGB"))
        negative = self.transform(Image.open(n_path).convert("RGB"))

        return anchor, positive, negative


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("=== dataset.py smoke test ===")
    ds = TripletFaceDataset(split="train", augment=False)
    print(f"Dataset length : {len(ds)}")

    a, p, n = ds[0]
    print(f"Anchor  shape  : {a.shape}   dtype={a.dtype}")
    print(f"Positive shape : {p.shape}")
    print(f"Negative shape : {n.shape}")

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch_a, batch_p, batch_n = next(iter(loader))
    print(f"Batch shapes   : {batch_a.shape}, {batch_p.shape}, {batch_n.shape}")
    print("OK - dataset.py passed.")
