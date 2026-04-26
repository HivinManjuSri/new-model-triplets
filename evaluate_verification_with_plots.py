"""
evaluate_verification_with_plots.py
---------------------------------
Comprehensive evaluation script for a face verification model.

What this script does:
1) Uses the TEST dataset split (TripletFaceDataset(split="test")).
2) Computes binary verification metrics from pair distances:
   - Accuracy, Precision, Recall, Specificity
   - Confusion Matrix (TN, FP, FN, TP)
   - FAR, FRR, EER
3) Computes ROC curve and AUC.
4) Saves graphs:
   - ROC curve
   - FAR/FRR vs threshold curve
   - Genuine vs impostor distance histogram
5) Measures model response speed:
   - Single-image embedding latency (ms)
   - Pair verification latency (ms)
   - Throughput (images/sec)

Pair construction from each triplet (A, P, N):
- Genuine pair : (A, P) with label 1
- Impostor pair: (A, N) with label 0

Decision rule at threshold t:
- predict same person if distance < t

Usage examples:
python evaluate_verification_with_plots.py
python evaluate_verification_with_plots.py --checkpoint checkpoints/best_model.pt --batch_size 16
python evaluate_verification_with_plots.py --plot_dir plots --save_json reports/test_eval_report.json
"""

import argparse
import json
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
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
    tn: int
    fp: int
    fn: int
    tp: int


def sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA clock for accurate timing when running on GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load checkpoint to model and switch to eval mode."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def collect_test_distances(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect d(A,P) and d(A,N) distances from the TEST split."""
    pos_dist = []
    neg_dist = []

    model.eval()
    with torch.no_grad():
        for anchor, positive, negative in tqdm(loader, desc="Collecting TEST distances"):
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            d_ap = F.pairwise_distance(emb_a, emb_p, p=2)
            d_an = F.pairwise_distance(emb_a, emb_n, p=2)

            pos_dist.extend(d_ap.detach().cpu().numpy())
            neg_dist.extend(d_an.detach().cpu().numpy())

    return np.asarray(pos_dist, dtype=np.float64), np.asarray(neg_dist, dtype=np.float64)


def build_binary_arrays(pos_dist: np.ndarray, neg_dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct labels and distance arrays for binary verification evaluation."""
    labels = np.concatenate([
        np.ones(len(pos_dist), dtype=np.int64),
        np.zeros(len(neg_dist), dtype=np.int64),
    ])
    distances = np.concatenate([pos_dist, neg_dist])
    return labels, distances


def confusion_from_threshold(labels: np.ndarray, distances: np.ndarray, threshold: float) -> Confusion:
    """Compute TN/FP/FN/TP for rule: same if distance < threshold."""
    pred_same = distances < threshold
    true_same = labels == 1

    tp = int(np.sum(pred_same & true_same))
    tn = int(np.sum((~pred_same) & (~true_same)))
    fp = int(np.sum(pred_same & (~true_same)))
    fn = int(np.sum((~pred_same) & true_same))
    return Confusion(tn=tn, fp=fp, fn=fn, tp=tp)


def metrics_from_confusion(cm: Confusion) -> dict:
    """Compute classification + verification rates from confusion matrix."""
    total = cm.tp + cm.tn + cm.fp + cm.fn

    accuracy = (cm.tp + cm.tn) / (total + EPS)
    precision = cm.tp / (cm.tp + cm.fp + EPS)
    recall = cm.tp / (cm.tp + cm.fn + EPS)
    specificity = cm.tn / (cm.tn + cm.fp + EPS)

    far = cm.fp / (cm.fp + cm.tn + EPS)
    frr = cm.fn / (cm.fn + cm.tp + EPS)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "far": float(far),
        "frr": float(frr),
    }


def compute_triplet_accuracy(pos_dist: np.ndarray, neg_dist: np.ndarray, margin: float) -> float:
    """Triplet correctness fraction for d(A,P) + margin < d(A,N)."""
    return float(np.mean((pos_dist + margin) < neg_dist))


def threshold_sweep(labels: np.ndarray, distances: np.ndarray, num_thresholds: int) -> dict:
    """
    Sweep thresholds and compute ROC/FAR/FRR points.

    ROC definition:
    - TPR = Recall = TP / (TP + FN)
    - FPR = FP / (FP + TN)

    Since lower distance means more likely genuine, we threshold distances.
    """
    dmin = float(np.min(distances))
    dmax = float(np.max(distances))
    thresholds = np.linspace(dmin, dmax, num=num_thresholds)

    rows = []
    best_acc = -1.0
    best_acc_threshold = thresholds[0]

    for th in thresholds:
        cm = confusion_from_threshold(labels, distances, float(th))
        m = metrics_from_confusion(cm)
        tpr = m["recall"]
        fpr = 1.0 - m["specificity"]

        row = {
            "threshold": float(th),
            "tn": cm.tn,
            "fp": cm.fp,
            "fn": cm.fn,
            "tp": cm.tp,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "far": m["far"],
            "frr": m["frr"],
            "tpr": float(tpr),
            "fpr": float(fpr),
        }
        rows.append(row)

        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_acc_threshold = float(th)

    # EER is where FAR and FRR are equal (or closest in a discrete sweep).
    diff = np.array([abs(r["far"] - r["frr"]) for r in rows], dtype=np.float64)
    eer_idx = int(np.argmin(diff))
    eer_threshold = rows[eer_idx]["threshold"]
    eer = 0.5 * (rows[eer_idx]["far"] + rows[eer_idx]["frr"])

    # ROC AUC via trapezoidal integration over FPR-sorted points.
    fpr = np.array([r["fpr"] for r in rows], dtype=np.float64)
    tpr = np.array([r["tpr"] for r in rows], dtype=np.float64)
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]
    # NumPy API compatibility: prefer trapezoid, fall back to trapz.
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
    else:
        auc = float(np.trapz(tpr_sorted, fpr_sorted))

    return {
        "rows": rows,
        "best_accuracy": float(best_acc),
        "best_accuracy_threshold": float(best_acc_threshold),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "auc": auc,
    }


def save_roc_plot(rows: list[dict], auc: float, out_path: str) -> None:
    fpr = np.array([r["fpr"] for r in rows], dtype=np.float64)
    tpr = np.array([r["tpr"] for r in rows], dtype=np.float64)
    order = np.argsort(fpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr[order], tpr[order], linewidth=2.0, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_far_frr_plot(rows: list[dict], eer: float, eer_threshold: float, out_path: str) -> None:
    thresholds = np.array([r["threshold"] for r in rows], dtype=np.float64)
    far = np.array([r["far"] for r in rows], dtype=np.float64)
    frr = np.array([r["frr"] for r in rows], dtype=np.float64)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far, linewidth=2.0, label="FAR")
    plt.plot(thresholds, frr, linewidth=2.0, label="FRR")
    plt.axvline(eer_threshold, linestyle="--", linewidth=1.2, label=f"EER threshold={eer_threshold:.4f}")
    plt.axhline(eer, linestyle=":", linewidth=1.2, label=f"EER={100*eer:.2f}%")
    plt.xlabel("Distance Threshold")
    plt.ylabel("Rate")
    plt.title("FAR / FRR vs Threshold")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_distance_histogram(pos_dist: np.ndarray, neg_dist: np.ndarray, threshold: float, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(pos_dist, bins=40, alpha=0.65, density=True, label="Genuine pairs (A,P)")
    plt.hist(neg_dist, bins=40, alpha=0.65, density=True, label="Impostor pairs (A,N)")
    plt.axvline(threshold, linestyle="--", linewidth=1.4, label=f"Threshold={threshold:.4f}")
    plt.xlabel("L2 Distance")
    plt.ylabel("Density")
    plt.title("Distance Distribution")
    plt.legend(loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def measure_latency(
    model: torch.nn.Module,
    dataset: TripletFaceDataset,
    device: torch.device,
    threshold: float,
    samples: int,
    warmup: int,
) -> dict:
    """
    Measure model response speed.

    We report:
    - embedding latency: one forward pass for one image
    - pair verification latency: two forwards + distance + threshold decision
    """
    samples = min(samples, len(dataset))

    embedding_times_ms = []
    pair_times_ms = []

    # Warm-up runs to avoid first-run setup overhead in timing.
    with torch.no_grad():
        for i in range(min(warmup, samples)):
            anchor, positive, _ = dataset[i]
            a = anchor.unsqueeze(0).to(device)
            p = positive.unsqueeze(0).to(device)
            _ = model(a)
            _ = model(p)
        sync_if_cuda(device)

    with torch.no_grad():
        for i in range(samples):
            anchor, positive, _ = dataset[i]

            a = anchor.unsqueeze(0).to(device)
            p = positive.unsqueeze(0).to(device)

            # Single image embedding latency
            sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = model(a)
            sync_if_cuda(device)
            t1 = time.perf_counter()
            embedding_times_ms.append((t1 - t0) * 1000.0)

            # Pair verification latency (end-to-end for one pair)
            sync_if_cuda(device)
            t2 = time.perf_counter()
            emb_a = model(a)
            emb_p = model(p)
            dist = F.pairwise_distance(emb_a, emb_p, p=2).item()
            _same = dist < threshold
            sync_if_cuda(device)
            t3 = time.perf_counter()
            pair_times_ms.append((t3 - t2) * 1000.0)

    emb = np.asarray(embedding_times_ms, dtype=np.float64)
    pair = np.asarray(pair_times_ms, dtype=np.float64)

    # Throughput estimate from average embedding latency.
    emb_mean_ms = float(np.mean(emb))
    throughput_img_per_sec = float(1000.0 / (emb_mean_ms + EPS))

    return {
        "samples": int(samples),
        "embedding_latency_ms": {
            "mean": float(np.mean(emb)),
            "median": float(np.median(emb)),
            "p95": percentile(emb, 95),
            "min": float(np.min(emb)),
            "max": float(np.max(emb)),
        },
        "pair_verification_latency_ms": {
            "mean": float(np.mean(pair)),
            "median": float(np.median(pair)),
            "p95": percentile(pair, 95),
            "min": float(np.min(pair)),
            "max": float(np.max(pair)),
        },
        "throughput_images_per_sec": throughput_img_per_sec,
    }


def pct(x: float) -> str:
    return f"{100.0*x:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test-set verification metrics + ROC/AUC plots + speed")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--num_thresholds", type=int, default=800)
    parser.add_argument("--threshold", type=float, default=None, help="Manual threshold; default picks best-accuracy threshold")
    parser.add_argument("--plot_dir", type=str, default="plots")
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--latency_samples", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Explicitly use TEST split as requested.
    print("Loading dataset split: TEST")
    test_ds = TripletFaceDataset(split="test", augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Loading model checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    pos_dist, neg_dist = collect_test_distances(model, test_loader, device)
    labels, distances = build_binary_arrays(pos_dist, neg_dist)

    sweep = threshold_sweep(labels, distances, num_thresholds=args.num_thresholds)
    chosen_threshold = args.threshold if args.threshold is not None else sweep["best_accuracy_threshold"]

    cm = confusion_from_threshold(labels, distances, chosen_threshold)
    m = metrics_from_confusion(cm)
    tri_acc = compute_triplet_accuracy(pos_dist, neg_dist, margin=args.margin)

    os.makedirs(args.plot_dir, exist_ok=True)
    roc_path = os.path.join(args.plot_dir, "roc_curve.png")
    far_frr_path = os.path.join(args.plot_dir, "far_frr_curve.png")
    hist_path = os.path.join(args.plot_dir, "distance_histogram.png")

    save_roc_plot(sweep["rows"], sweep["auc"], roc_path)
    save_far_frr_plot(sweep["rows"], sweep["eer"], sweep["eer_threshold"], far_frr_path)
    save_distance_histogram(pos_dist, neg_dist, chosen_threshold, hist_path)

    speed = measure_latency(
        model=model,
        dataset=test_ds,
        device=device,
        threshold=chosen_threshold,
        samples=args.latency_samples,
        warmup=args.warmup,
    )

    print("\n" + "=" * 76)
    print("TEST DATASET EVALUATION RESULTS")
    print("=" * 76)
    print(f"Test triplets: {len(test_ds)}")
    print(f"Verification pairs: {len(labels)}")

    print("\nClassification metrics at chosen threshold")
    print(f"Threshold:   {chosen_threshold:.6f}")
    print(f"Confusion:   TN={cm.tn}, FP={cm.fp}, FN={cm.fn}, TP={cm.tp}")
    print(f"Accuracy:    {pct(m['accuracy'])}")
    print(f"Precision:   {pct(m['precision'])}")
    print(f"Recall:      {pct(m['recall'])}")
    print(f"Specificity: {pct(m['specificity'])}")

    print("\nVerification metrics")
    print(f"FAR @ chosen threshold: {pct(m['far'])}")
    print(f"FRR @ chosen threshold: {pct(m['frr'])}")
    print(f"EER:                    {pct(sweep['eer'])}")
    print(f"EER threshold:          {sweep['eer_threshold']:.6f}")
    print(f"ROC AUC:                {sweep['auc']:.6f}")

    print("\nTriplet metric")
    print(f"Triplet accuracy (margin={args.margin}): {pct(tri_acc)}")

    print("\nModel response speed")
    print(f"Samples used for latency: {speed['samples']}")
    print(
        "Embedding latency (ms): "
        f"mean={speed['embedding_latency_ms']['mean']:.3f}, "
        f"median={speed['embedding_latency_ms']['median']:.3f}, "
        f"p95={speed['embedding_latency_ms']['p95']:.3f}"
    )
    print(
        "Pair verify latency (ms): "
        f"mean={speed['pair_verification_latency_ms']['mean']:.3f}, "
        f"median={speed['pair_verification_latency_ms']['median']:.3f}, "
        f"p95={speed['pair_verification_latency_ms']['p95']:.3f}"
    )
    print(f"Estimated throughput (single-image): {speed['throughput_images_per_sec']:.2f} images/sec")

    print("\nSaved plots")
    print(f"- {roc_path}")
    print(f"- {far_frr_path}")
    print(f"- {hist_path}")

    report = {
        "uses_dataset_split": "test",
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_triplets": int(len(test_ds)),
        "num_pairs": int(len(labels)),
        "chosen_threshold": float(chosen_threshold),
        "best_accuracy_threshold": float(sweep["best_accuracy_threshold"]),
        "best_accuracy": float(sweep["best_accuracy"]),
        "confusion_matrix": {
            "tn": cm.tn,
            "fp": cm.fp,
            "fn": cm.fn,
            "tp": cm.tp,
        },
        "classification_metrics": m,
        "verification_metrics": {
            "far": m["far"],
            "frr": m["frr"],
            "eer": float(sweep["eer"]),
            "eer_threshold": float(sweep["eer_threshold"]),
            "auc": float(sweep["auc"]),
        },
        "triplet_accuracy": float(tri_acc),
        "speed": speed,
        "plots": {
            "roc_curve": roc_path,
            "far_frr_curve": far_frr_path,
            "distance_histogram": hist_path,
        },
    }

    if args.save_json:
        out_dir = os.path.dirname(args.save_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report JSON: {args.save_json}")


if __name__ == "__main__":
    main()
