"""
model.py
--------
Siamese / one-shot face embedding network.

Architecture:
    ResNet-50 backbone  (pretrained on ImageNet, final FC removed)
        ↓  2048-d feature vector
    Embedding head:
        Linear(2048 → 512) → BN → ReLU → Dropout(0.3)
        Linear(512  → 128)
        L2-normalise                   ← unit-sphere embedding
        ↓  128-d unit embedding

The same network is used for all three branches (A / P / N) — this is
the Siamese weight-sharing property.

Usage:
    model = FaceEmbeddingNet()
    # single image batch
    emb = model(images)          # shape: (B, 128)

    # triplet batch
    ea, ep, en = model(anchor), model(positive), model(negative)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


EMBEDDING_DIM = 128


class FaceEmbeddingNet(nn.Module):
    """ResNet-50 backbone with a lightweight embedding head."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, pretrained: bool = True):
        super().__init__()

        # ── Backbone ────────────────────────────────────────────────────────
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original classification head (FC layer)
        # backbone.fc is Linear(2048, 1000) — we replace it with Identity
        in_features = backbone.fc.in_features   # 2048
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # ── Embedding head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  — ImageNet-normalised face images
        Returns:
            emb: (B, embedding_dim)  — L2-normalised unit embeddings
        """
        features = self.backbone(x)          # (B, 2048)
        emb = self.head(features)            # (B, 128)
        emb = F.normalize(emb, p=2, dim=1)  # L2-normalise → unit sphere
        return emb


# ── Convenience factory ─────────────────────────────────────────────────────

def build_model(embedding_dim: int = EMBEDDING_DIM,
                pretrained: bool = True) -> FaceEmbeddingNet:
    return FaceEmbeddingNet(embedding_dim=embedding_dim, pretrained=pretrained)


if __name__ == "__main__":
    # Quick sanity check
    model = build_model()
    dummy = torch.randn(4, 3, 112, 112)
    out = model(dummy)
    print(f"Output shape : {out.shape}")          # (4, 128)
    print(f"L2 norms (should be 1.0): {out.norm(dim=1)}")  # all 1.0
