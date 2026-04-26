"""
analyze_errors.py
-----------------
Analyze which triplets the model gets wrong to identify patterns.

Usage:
    python analyze_errors.py
    python analyze_errors.py --threshold 0.8
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from dataset import TripletFaceDataset
from model import build_model


def analyze_errors(checkpoint_path='checkpoints/best_model.pt', threshold=0.8):
    """Find and categorize errors."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    
    # Load test set
    test_dataset = TripletFaceDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    false_negatives = []  # Same person but distance > threshold
    false_positives = []  # Different person but distance < threshold
    
    print(f"Analyzing errors with threshold={threshold}...\n")
    
    with torch.no_grad():
        for idx, (anchor, positive, negative) in enumerate(test_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Get embeddings
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # Compute distances
            pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2).item()
            neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2).item()
            
            # Check for errors
            if pos_dist > threshold:  # False Negative
                false_negatives.append({
                    'idx': idx,
                    'pos_dist': pos_dist,
                    'neg_dist': neg_dist,
                    'severity': pos_dist - threshold
                })
            
            if neg_dist < threshold:  # False Positive
                false_positives.append({
                    'idx': idx,
                    'pos_dist': pos_dist,
                    'neg_dist': neg_dist,
                    'severity': threshold - neg_dist
                })
    
    # Print results
    print("="*70)
    print(f"ERROR ANALYSIS (Threshold: {threshold})")
    print("="*70)
    print(f"Total test triplets: {len(test_dataset)}")
    print(f"False Negatives:     {len(false_negatives)} ({len(false_negatives)/len(test_dataset)*100:.2f}%)")
    print(f"False Positives:     {len(false_positives)} ({len(false_positives)/len(test_dataset)*100:.2f}%)")
    print()
    
    # False Negatives (same person rejected)
    if false_negatives:
        print("\n" + "="*70)
        print("FALSE NEGATIVES (Same person but distance > threshold)")
        print("="*70)
        fn_sorted = sorted(false_negatives, key=lambda x: x['severity'], reverse=True)
        
        print(f"{'Triplet':<12} {'Pos Dist':<12} {'Neg Dist':<12} {'Severity':<12}")
        print("-"*70)
        for err in fn_sorted[:20]:  # Show worst 20
            print(f"{err['idx']:<12} {err['pos_dist']:<12.4f} {err['neg_dist']:<12.4f} {err['severity']:<12.4f}")
        
        pos_dists = [e['pos_dist'] for e in false_negatives]
        print(f"\nFalse Negative Statistics:")
        print(f"  Mean distance:   {np.mean(pos_dists):.4f}")
        print(f"  Median distance: {np.median(pos_dists):.4f}")
        print(f"  Max distance:    {np.max(pos_dists):.4f}")
    
    # False Positives (different person accepted)
    if false_positives:
        print("\n" + "="*70)
        print("FALSE POSITIVES (Different person but distance < threshold)")
        print("="*70)
        fp_sorted = sorted(false_positives, key=lambda x: x['severity'], reverse=True)
        
        print(f"{'Triplet':<12} {'Pos Dist':<12} {'Neg Dist':<12} {'Severity':<12}")
        print("-"*70)
        for err in fp_sorted[:20]:  # Show worst 20
            print(f"{err['idx']:<12} {err['pos_dist']:<12.4f} {err['neg_dist']:<12.4f} {err['severity']:<12.4f}")
        
        neg_dists = [e['neg_dist'] for e in false_positives]
        print(f"\nFalse Positive Statistics:")
        print(f"  Mean distance:   {np.mean(neg_dists):.4f}")
        print(f"  Median distance: {np.median(neg_dists):.4f}")
        print(f"  Min distance:    {np.min(neg_dists):.4f}")
    
    print("\n" + "="*70)
    print("\nTIP: Inspect the triplet folders for false positives/negatives")
    print("     to understand what types of faces are challenging.")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()
    
    analyze_errors(args.checkpoint, args.threshold)
