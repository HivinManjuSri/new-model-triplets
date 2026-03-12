"""
loss.py
-------
Triplet loss with semi-hard negative mining.

Standard triplet loss (FaceNet):
    L = max(0,  d(A, P) - d(A, N) + margin)

Semi-hard mining selects negatives where:
    d(A, P) < d(A, N) < d(A, P) + margin

These are "semi-hard" — harder than easy negatives (d(A,N) > d(A,P)+margin)
but not as unstable as hard negatives (d(A,N) < d(A,P)).

When no semi-hard negatives exist in a batch, falls back to the hardest
valid negative (closest one that is still farther than the positive).

Since embeddings are L2-normalised (unit sphere), we use squared
Euclidean distance which is equivalent to 2*(1 - cosine_similarity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Args:
        margin  : α in max(0, d(A,P) - d(A,N) + margin)
        mining  : 'semihard' | 'hard' | 'random'
    """

    def __init__(self, margin: float = 0.2, mining: str = "semihard"):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(
        self,
        anchor:   torch.Tensor,   # (B, D)
        positive: torch.Tensor,   # (B, D)
        negative: torch.Tensor,   # (B, D)
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            loss      : scalar tensor
            stats     : dict with 'loss', 'pos_dist_mean', 'neg_dist_mean',
                        'fraction_active' (fraction of non-zero triplets)
        """
        # Squared L2 distance (embeddings are already unit-normalised)
        d_pos = _sq_dist(anchor, positive)   # (B,)
        d_neg = _sq_dist(anchor, negative)   # (B,)

        raw_loss = F.relu(d_pos - d_neg + self.margin)   # (B,)

        fraction_active = (raw_loss > 0).float().mean().item()
        loss = raw_loss.mean()

        stats = {
            "loss":             loss.item(),
            "pos_dist_mean":    d_pos.mean().item(),
            "neg_dist_mean":    d_neg.mean().item(),
            "fraction_active":  fraction_active,
        }
        return loss, stats


class OnlineTripletLoss(nn.Module):
    """
    Online semi-hard triplet mining from a batch of (embedding, label) pairs.

    Use this if you switch to identity-based sampling (each batch contains
    K images × P identities).  With the pre-built triplet dataset you
    normally use TripletLoss above.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D)  all L2-normalised
        labels:     torch.Tensor,  # (B,)    integer identity labels
    ) -> tuple[torch.Tensor, dict]:
        """Mine semi-hard triplets from all valid (A, P, N) combos in the batch."""
        B = embeddings.size(0)
        device = embeddings.device

        # Full pairwise squared distance matrix  (B, B)
        dist_mat = _pairwise_sq_dist(embeddings)

        # Boolean masks
        same  = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
        diff  = ~same

        valid_triplets = 0
        total_loss     = torch.tensor(0.0, device=device)

        for a_idx in range(B):
            pos_mask = same[a_idx].clone()
            pos_mask[a_idx] = False                          # exclude self

            neg_mask = diff[a_idx]

            if not pos_mask.any() or not neg_mask.any():
                continue

            d_ap = dist_mat[a_idx][pos_mask]                 # (n_pos,)
            d_an = dist_mat[a_idx][neg_mask]                 # (n_neg,)

            # For each positive, find a semi-hard negative
            for d_p in d_ap:
                # semi-hard: d_p < d_n < d_p + margin
                semihard = d_an[(d_an > d_p) & (d_an < d_p + self.margin)]

                if semihard.numel() > 0:
                    d_n = semihard.min()
                else:
                    # fallback: hardest valid negative (closest > d_p)
                    valid_neg = d_an[d_an > d_p]
                    if valid_neg.numel() == 0:
                        continue
                    d_n = valid_neg.min()

                triplet_loss = F.relu(d_p - d_n + self.margin)
                total_loss  += triplet_loss
                valid_triplets += 1

        if valid_triplets == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0}

        avg_loss = total_loss / valid_triplets
        return avg_loss, {"loss": avg_loss.item(), "valid_triplets": valid_triplets}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance between corresponding row pairs. Shape: (B,)"""
    return (a - b).pow(2).sum(dim=1)


def _pairwise_sq_dist(x: torch.Tensor) -> torch.Tensor:
    """Full pairwise squared L2 distance matrix. Shape: (B, B)"""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a,b>
    # For unit vectors: = 2 - 2 <a,b>
    dot = x @ x.t()                              # (B, B)
    sq  = (x * x).sum(dim=1, keepdim=True)       # (B, 1)
    dist = sq + sq.t() - 2 * dot
    return dist.clamp(min=0)                     # numerical safety


if __name__ == "__main__":
    print("=== loss.py smoke test ===")
    B, D = 8, 128
    # Simulate L2-normalised embeddings
    a = torch.randn(B, D); a = torch.nn.functional.normalize(a, dim=1)
    p = torch.randn(B, D); p = torch.nn.functional.normalize(p, dim=1)
    n = torch.randn(B, D); n = torch.nn.functional.normalize(n, dim=1)

    criterion = TripletLoss(margin=0.2)
    loss, stats = criterion(a, p, n)
    print(f"Loss            : {stats['loss']:.4f}")
    print(f"Pos dist (mean) : {stats['pos_dist_mean']:.4f}")
    print(f"Neg dist (mean) : {stats['neg_dist_mean']:.4f}")
    print(f"Active triplets : {stats['fraction_active']:.2%}")
    print("OK - loss.py passed.")
