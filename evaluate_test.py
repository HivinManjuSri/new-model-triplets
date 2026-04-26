"""
evaluate_test.py
----------------
Evaluate the trained model on the test set.
Computes accuracy, distance metrics, and face verification metrics.

Usage:
    python evaluate_test.py
    python evaluate_test.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import TripletFaceDataset
from model import build_model
from loss import TripletLoss


def compute_metrics(pos_distances, neg_distances, margin=0.2):
    """
    Compute evaluation metrics for face verification.
    
    Args:
        pos_distances: List of distances for positive pairs (same person)
        neg_distances: List of distances for negative pairs (different person)
        margin: Triplet loss margin
    
    Returns:
        Dictionary of metrics
    """
    pos_distances = np.array(pos_distances)
    neg_distances = np.array(neg_distances)
    
    # Triplet accuracy: percentage where d(A,P) + margin < d(A,N)
    triplet_correct = (pos_distances + margin) < neg_distances
    triplet_accuracy = triplet_correct.mean() * 100
    
    # Distance statistics
    metrics = {
        'triplet_accuracy': triplet_accuracy,
        'pos_dist_mean': pos_distances.mean(),
        'pos_dist_std': pos_distances.std(),
        'pos_dist_median': np.median(pos_distances),
        'neg_dist_mean': neg_distances.mean(),
        'neg_dist_std': neg_distances.std(),
        'neg_dist_median': np.median(neg_distances),
        'margin_achieved': neg_distances.mean() - pos_distances.mean(),
    }
    
    # Face verification accuracy at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print("\n" + "="*70)
    print("FACE VERIFICATION ACCURACY AT DIFFERENT THRESHOLDS")
    print("="*70)
    print(f"{'Threshold':<12} {'TP (same)':<12} {'TN (diff)':<12} {'Accuracy':<12}")
    print("-"*70)
    
    for threshold in thresholds:
        # True Positive: same person, distance < threshold
        tp = (pos_distances < threshold).sum()
        tp_rate = (tp / len(pos_distances)) * 100
        
        # True Negative: different person, distance >= threshold
        tn = (neg_distances >= threshold).sum()
        tn_rate = (tn / len(neg_distances)) * 100
        
        # Overall accuracy
        accuracy = ((tp + tn) / (len(pos_distances) + len(neg_distances))) * 100
        
        print(f"{threshold:<12.2f} {tp_rate:<12.1f}% {tn_rate:<12.1f}% {accuracy:<12.1f}%")
        
        metrics[f'accuracy_at_{threshold}'] = accuracy
    
    return metrics


def evaluate(model, dataloader, device, margin=0.2):
    """Evaluate model on a dataset."""
    model.eval()
    
    pos_distances = []
    neg_distances = []
    total_loss = 0.0
    
    loss_fn = TripletLoss(margin=margin)
    
    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc="Evaluating"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Get embeddings
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # Compute distances
            pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
            neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
            
            pos_distances.extend(pos_dist.cpu().numpy())
            neg_distances.extend(neg_dist.cpu().numpy())
            
            # Compute loss
            loss, _ = loss_fn(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(pos_distances, neg_distances, margin)
    metrics['loss'] = avg_loss
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Triplet loss margin')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print(f"\nLoading test dataset...")
    test_dataset = TripletFaceDataset(split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Test set: {len(test_dataset)} triplets")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = build_model(pretrained=False).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    
    if 'epoch' in ckpt:
        print(f"Checkpoint from epoch: {ckpt['epoch']}")
    if 'val_loss' in ckpt:
        print(f"Validation loss: {ckpt['val_loss']:.6f}")
    
    # Evaluate
    print(f"\n{'='*70}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*70}")
    
    metrics = evaluate(model, test_loader, device, args.margin)
    
    # Print results
    print(f"\n{'='*70}")
    print("TEST SET RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss:              {metrics['loss']:.6f}")
    print(f"Triplet Accuracy:       {metrics['triplet_accuracy']:.2f}%")
    print(f"\nPositive Distance (same person):")
    print(f"  Mean:    {metrics['pos_dist_mean']:.4f}")
    print(f"  Std:     {metrics['pos_dist_std']:.4f}")
    print(f"  Median:  {metrics['pos_dist_median']:.4f}")
    print(f"\nNegative Distance (different person):")
    print(f"  Mean:    {metrics['neg_dist_mean']:.4f}")
    print(f"  Std:     {metrics['neg_dist_std']:.4f}")
    print(f"  Median:  {metrics['neg_dist_median']:.4f}")
    print(f"\nMargin Achieved:        {metrics['margin_achieved']:.4f}")
    print(f"Required Margin:        {args.margin:.4f}")
    print(f"Margin Ratio:           {metrics['margin_achieved']/args.margin:.2f}x")
    print(f"{'='*70}")
    
    # Find best threshold
    best_threshold = None
    best_accuracy = 0
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        acc = metrics[f'accuracy_at_{threshold}']
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold
    
    print(f"\n🎯 Best threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.2f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
