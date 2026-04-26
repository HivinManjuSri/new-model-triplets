"""
evaluate_verification_metrics.py
------------------------------
Paper-style evaluation script for face verification.

This script reports:
1) Classification metrics (from a chosen threshold):
   - Accuracy
   - Precision
   - Recall (Sensitivity / TPR)
   - Specificity (TNR)
   - Confusion Matrix (TN, FP, FN, TP)

2) Verification metrics (from threshold sweep):
   - FAR (False Accept Rate)
   - FRR (False Reject Rate)
   - EER (Equal Error Rate)

3) Triplet-style metric:
   - Triplet accuracy: d(A,P) + margin < d(A,N)

Why this works:
- For each triplet (A, P, N), we create two verification pairs:
  - Genuine pair  : (A, P), label=1
  - Impostor pair : (A, N), label=0
- Distances are computed in embedding space (L2 distance).
- Decision rule is: predict same person if distance < threshold.

Usage examples:
    python evaluate_verification_metrics.py
    python evaluate_verification_metrics.py --checkpoint checkpoints/best_model.pt --batch_size 16
    python evaluate_verification_metrics.py --threshold 0.8 --save_json metrics_report.json
"""

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TripletFaceDataset
from model import build_model


EPS = 1e-12


@dataclass
class Confusion:
    """Container for confusion matrix counts."""
    tn: int
    fp: int
    fn: int
    tp: int


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model weights from checkpoint."""
    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def collect_distances(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect distances for genuine (A,P) and impostor (A,N) pairs.

    Returns:
        pos_distances: distances for same-identity pairs
        neg_distances: distances for different-identity pairs
    """
    pos_distances = []
    neg_distances = []

    model.eval()
    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc="Extracting distances"):
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
            d_an = F.pairwise_distance(emb_a, emb_n, p=2)

            pos_distances.extend(d_ap.detach().cpu().numpy())
            neg_distances.extend(d_an.detach().cpu().numpy())

    return np.asarray(pos_distances, dtype=np.float64), np.asarray(neg_distances, dtype=np.float64)


def build_binary_eval_arrays(
    pos_distances: np.ndarray,
    neg_distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build arrays for binary verification evaluation.

    labels: 1 for same person, 0 for different person
    scores: L2 distance (lower means more likely same)
    """
    labels = np.concatenate([
        np.ones(len(pos_distances), dtype=np.int64),
        np.zeros(len(neg_distances), dtype=np.int64),
    ])
    distances = np.concatenate([pos_distances, neg_distances])
    return labels, distances


def confusion_from_threshold(labels: np.ndarray, distances: np.ndarray, threshold: float) -> Confusion:
    """
    Compute confusion matrix at one threshold.

    Decision rule:
        predicted_same = distance < threshold
    """
    pred_same = distances < threshold
    true_same = labels == 1

    tp = int(np.sum(pred_same & true_same))
    tn = int(np.sum((~pred_same) & (~true_same)))
    fp = int(np.sum(pred_same & (~true_same)))
    fn = int(np.sum((~pred_same) & true_same))

    return Confusion(tn=tn, fp=fp, fn=fn, tp=tp)


def metrics_from_confusion(cm: Confusion) -> dict:
    """Compute classification and verification metrics from confusion counts."""
    total = cm.tp + cm.tn + cm.fp + cm.fn

    accuracy = (cm.tp + cm.tn) / (total + EPS)
    precision = cm.tp / (cm.tp + cm.fp + EPS)
    recall = cm.tp / (cm.tp + cm.fn + EPS)          # sensitivity / TPR
    specificity = cm.tn / (cm.tn + cm.fp + EPS)     # TNR

    # Verification rates commonly used in biometrics
    far = cm.fp / (cm.fp + cm.tn + EPS)             # false accept among impostors
    frr = cm.fn / (cm.fn + cm.tp + EPS)             # false reject among genuine pairs

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "far": far,
        "frr": frr,
    }


def threshold_sweep(labels: np.ndarray, distances: np.ndarray, num_thresholds: int = 600) -> dict:
    """
    Sweep thresholds to build FAR/FRR curves and find EER.

    EER is where FAR and FRR are equal (or closest in discrete sweep).
    """
    d_min = float(np.min(distances))
    d_max = float(np.max(distances))
    thresholds = np.linspace(d_min, d_max, num=num_thresholds)

    rows = []
    best_acc = -1.0
    best_acc_threshold = thresholds[0]

    for th in thresholds:
        cm = confusion_from_threshold(labels, distances, float(th))
        m = metrics_from_confusion(cm)
        row = {
            "threshold": float(th),
            "tn": cm.tn,
            "fp": cm.fp,
            "fn": cm.fn,
            "tp": cm.tp,
            **m,
        }
        rows.append(row)

        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_acc_threshold = float(th)

    # Approximate EER by choosing threshold where |FAR - FRR| is minimal
    abs_gap = np.array([abs(r["far"] - r["frr"]) for r in rows], dtype=np.float64)
    idx_eer = int(np.argmin(abs_gap))
    eer_threshold = rows[idx_eer]["threshold"]
    eer = 0.5 * (rows[idx_eer]["far"] + rows[idx_eer]["frr"])

    return {
        "rows": rows,
        "best_accuracy_threshold": best_acc_threshold,
        "best_accuracy": best_acc,
        "eer_threshold": eer_threshold,
        "eer": float(eer),
    }


def triplet_accuracy(pos_distances: np.ndarray, neg_distances: np.ndarray, margin: float) -> float:
    """Fraction of triplets satisfying d(A,P) + margin < d(A,N)."""
    ok = (pos_distances + margin) < neg_distances
    return float(np.mean(ok))


def format_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def print_metric_guide() -> None:
    """Quick human-readable definitions shown in terminal output."""
    print("\nMetric definitions:")
    print("- Accuracy    = (TP + TN) / (TP + TN + FP + FN)")
    print("- Precision   = TP / (TP + FP)")
    print("- Recall      = TP / (TP + FN)   (Sensitivity, TPR)")
    print("- Specificity = TN / (TN + FP)   (TNR)")
    print("- FAR         = FP / (FP + TN)   (false accepts among impostor pairs)")
    print("- FRR         = FN / (FN + TP)   (false rejects among genuine pairs)")
    print("- EER         = FAR = FRR at the operating point where they intersect")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive verification metrics evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet margin for triplet accuracy")
    parser.add_argument("--num_thresholds", type=int, default=600, help="Threshold points for FAR/FRR sweep")
    parser.add_argument("--threshold", type=float, default=None, help="Manual threshold for confusion metrics")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save report JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading test split...")
    test_ds = TripletFaceDataset(split="test", augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Test triplets: {len(test_ds)}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    pos_d, neg_d = collect_distances(model, test_loader, device)
    labels, distances = build_binary_eval_arrays(pos_d, neg_d)

    sweep = threshold_sweep(labels, distances, num_thresholds=args.num_thresholds)

    # If user does not force threshold, use best-accuracy threshold.
    chosen_threshold = args.threshold if args.threshold is not None else sweep["best_accuracy_threshold"]

    cm = confusion_from_threshold(labels, distances, chosen_threshold)
    cls = metrics_from_confusion(cm)
    tri_acc = triplet_accuracy(pos_d, neg_d, margin=args.margin)

    print("\n" + "=" * 72)
    print("CLASSIFICATION METRICS (AT CHOSEN THRESHOLD)")
    print("=" * 72)
    print(f"Threshold used: {chosen_threshold:.6f}")
    print(f"Confusion matrix counts: TN={cm.tn}, FP={cm.fp}, FN={cm.fn}, TP={cm.tp}")
    print(f"Accuracy:    {format_pct(cls['accuracy'])}")
    print(f"Precision:   {format_pct(cls['precision'])}")
    print(f"Recall:      {format_pct(cls['recall'])}")
    print(f"Specificity: {format_pct(cls['specificity'])}")

    print("\n" + "=" * 72)
    print("VERIFICATION METRICS (THRESHOLD SWEEP)")
    print("=" * 72)
    print(f"Best accuracy threshold: {sweep['best_accuracy_threshold']:.6f}")
    print(f"Best accuracy:           {format_pct(sweep['best_accuracy'])}")
    print(f"EER threshold:           {sweep['eer_threshold']:.6f}")
    print(f"EER:                     {format_pct(sweep['eer'])}")
    print(f"FAR @ chosen threshold:  {format_pct(cls['far'])}")
    print(f"FRR @ chosen threshold:  {format_pct(cls['frr'])}")

    print("\n" + "=" * 72)
    print("TRIPLET METRIC")
    print("=" * 72)
    print(f"Triplet accuracy (d(A,P)+margin < d(A,N), margin={args.margin}): {format_pct(tri_acc)}")

    print_metric_guide()

    if args.save_json:
        report = {
            "checkpoint": args.checkpoint,
            "device": str(device),
            "num_triplets": int(len(test_ds)),
            "num_pairs": int(len(labels)),
            "chosen_threshold": float(chosen_threshold),
            "confusion": {
                "tn": cm.tn,
                "fp": cm.fp,
                "fn": cm.fn,
                "tp": cm.tp,
            },
            "classification_metrics": cls,
            "verification_metrics": {
                "best_accuracy_threshold": sweep["best_accuracy_threshold"],
                "best_accuracy": sweep["best_accuracy"],
                "eer_threshold": sweep["eer_threshold"],
                "eer": sweep["eer"],
            },
            "triplet_accuracy": tri_acc,
        }

        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON report to: {args.save_json}")


if __name__ == "__main__":
    main()
