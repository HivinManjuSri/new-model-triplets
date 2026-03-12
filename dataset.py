"""
dataset.py
----------
PyTorch Dataset that reads pre-built triplets from the dataset.

Each triplet folder contains:
    A_XXXXX.ppm   Anchor  (identity XXXXX)
    P_XXXXX.ppm   Positive (same identity, different photo)
    N_YYYYY.ppm   Negative (different identity YYYYY)

Returns (anchor_tensor, positive_tensor, negative_tensor) ready for the
model.  Images are converted from PPM → RGB PIL → resized to INPUT_SIZE →
normalized to ImageNet stats.
"""

import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── Constants ───────────────────────────────────────────────────────────────
INPUT_SIZE   = 112
DATASET_DIR  = os.path.join(os.path.dirname(__file__), "triplets_dataset")
SPLITS_DIR   = os.path.join(os.path.dirname(__file__), "splits")

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


def _find_images(folder: str):
    """Return (anchor_path, positive_path, negative_path) inside a triplet folder."""
    anchor   = glob.glob(os.path.join(folder, "A_*.ppm"))
    positive = glob.glob(os.path.join(folder, "P_*.ppm"))
    negative = glob.glob(os.path.join(folder, "N_*.ppm"))

    if not (anchor and positive and negative):
        raise FileNotFoundError(
            f"Triplet folder {folder!r} is missing A/P/N images. "
            f"Found: {os.listdir(folder)}"
        )

    return anchor[0], positive[0], negative[0]


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

        split_file = os.path.join(SPLITS_DIR, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"{split_file} not found. Run split_dataset.py first."
            )

        with open(split_file) as f:
            self.triplet_names = [line.strip() for line in f if line.strip()]

        self.transform = _make_transform(augment)
        self.split = split
        print(f"[TripletFaceDataset] {split}: {len(self.triplet_names)} triplets loaded.")

    def __len__(self):
        return len(self.triplet_names)

    def __getitem__(self, idx: int):
        folder = os.path.join(DATASET_DIR, self.triplet_names[idx])
        a_path, p_path, n_path = _find_images(folder)

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
