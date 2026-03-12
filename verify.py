"""
verify.py
---------
Inference script: takes two face image paths and decides whether they
belong to the same person.

Usage:
    python verify.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg
    python verify.py --img1 face1.ppm --img2 face2.ppm --checkpoint checkpoints/best_model.pt
    python verify.py --img1 face1.ppm --img2 face2.ppm --threshold 0.5

Output:
    Similarity : 0.8731
    Distance   : 0.2538
    Decision   : SAME PERSON  (threshold=0.50)
"""

import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import build_model
from dataset import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


# ── Transform (no augmentation, same as val/test) ────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(pretrained=False).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path!r}\n"
            "Train the model first with:  python train.py"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)   # handle both raw state_dict and full checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def load_image(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _transform(img).unsqueeze(0).to(device)   # (1, 3, H, W)


def verify(img1_path: str,
           img2_path: str,
           checkpoint_path: str,
           threshold: float,
           device: torch.device) -> dict:
    """
    Returns a dict with:
        similarity  : cosine similarity  in [-1, 1]  (higher = more similar)
        distance    : L2 distance        in [0, 2]   (lower  = more similar)
        same_person : bool
    """
    model = load_model(checkpoint_path, device)

    with torch.no_grad():
        emb1 = model(load_image(img1_path, device))   # (1, 128)
        emb2 = model(load_image(img2_path, device))   # (1, 128)

    # Both embeddings are already L2-normalised by the model
    cosine_sim = F.cosine_similarity(emb1, emb2).item()
    l2_dist    = (emb1 - emb2).pow(2).sum().sqrt().item()
    same       = l2_dist < threshold

    return {
        "similarity":  cosine_sim,
        "distance":    l2_dist,
        "same_person": same,
        "threshold":   threshold,
    }


def parse_args():
    default_ckpt = os.path.join(os.path.dirname(__file__),
                                "checkpoints", "best_model.pt")
    p = argparse.ArgumentParser(description="Face verification inference")
    p.add_argument("--img1",       required=True,            help="Path to first face image")
    p.add_argument("--img2",       required=True,            help="Path to second face image")
    p.add_argument("--checkpoint", default=default_ckpt,     help="Path to trained model checkpoint")
    p.add_argument("--threshold",  type=float, default=0.9,  help="L2 distance threshold (lower=stricter)")
    p.add_argument("--device",     default="auto",           help="'cpu', 'cuda', or 'auto'")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device     : {device}")
    print(f"Image 1    : {args.img1}")
    print(f"Image 2    : {args.img2}")
    print(f"Checkpoint : {args.checkpoint}")
    print()

    result = verify(
        img1_path=args.img1,
        img2_path=args.img2,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device=device,
    )

    decision = "SAME PERSON" if result["same_person"] else "DIFFERENT PERSON"
    print(f"Similarity : {result['similarity']:.4f}")
    print(f"Distance   : {result['distance']:.4f}")
    print(f"Decision   : {decision}  (threshold={result['threshold']:.2f})")


if __name__ == "__main__":
    main()
