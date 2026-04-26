"""
inspect_triplet.py
------------------
Visualize a specific triplet to understand why it failed.

Usage:
    python inspect_triplet.py 511    # Worst false negative
    python inspect_triplet.py 994    # Worst false positive
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import TripletFaceDataset
from model import build_model


def visualize_triplet(triplet_idx, checkpoint_path='checkpoints/best_model.pt'):
    """Visualize a triplet and show model predictions."""
    
    # Load dataset
    dataset = TripletFaceDataset(split='test')
    
    if triplet_idx >= len(dataset):
        print(f"Error: Triplet index {triplet_idx} out of range (test set has {len(dataset)} triplets)")
        return
    
    # Get triplet tensors
    anchor_tensor, positive_tensor, negative_tensor = dataset[triplet_idx]
    
    # Get file paths from dataset
    a_path, p_path, n_path = dataset.triplets[triplet_idx]
    
    # Load images for display
    anchor_img = Image.open(a_path).convert('RGB')
    positive_img = Image.open(p_path).convert('RGB')
    negative_img = Image.open(n_path).convert('RGB')
    
    # Load model and compute distances
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    
    with torch.no_grad():
        anchor_emb = model(anchor_tensor.unsqueeze(0).to(device))
        positive_emb = model(positive_tensor.unsqueeze(0).to(device))
        negative_emb = model(negative_tensor.unsqueeze(0).to(device))
        
        pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2).item()
        neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2).item()
    
    # Extract person IDs from paths  (…/n000208/0194_01.jpg → n000208)
    anchor_id = Path(a_path).parent.name
    positive_id = Path(p_path).parent.name
    negative_id = Path(n_path).parent.name
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(anchor_img)
    axes[0].set_title(f'Anchor\nPerson ID: {anchor_id}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(positive_img)
    axes[1].set_title(f'Positive (Same Person)\nPerson ID: {positive_id}\nDistance: {pos_dist:.4f}', 
                      fontsize=14, fontweight='bold', 
                      color='green' if pos_dist < 0.8 else 'red')
    axes[1].axis('off')
    
    axes[2].imshow(negative_img)
    axes[2].set_title(f'Negative (Different Person)\nPerson ID: {negative_id}\nDistance: {neg_dist:.4f}', 
                      fontsize=14, fontweight='bold',
                      color='green' if neg_dist >= 0.8 else 'red')
    axes[2].axis('off')
    
    plt.suptitle(f'Triplet #{triplet_idx} - Test Set',
                 fontsize=16, fontweight='bold')
    
    # Add diagnostics
    fig.text(0.5, 0.02, 
             f'Threshold: 0.80 | Margin: {neg_dist - pos_dist:.4f} | '
             f'Status: {"CORRECT" if (pos_dist < 0.8 and neg_dist >= 0.8) else "ERROR"}',
             ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen' if (pos_dist < 0.8 and neg_dist >= 0.8) else 'lightcoral'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save image
    output_path = f'triplet_{triplet_idx}_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    plt.show()
    
    # Print detailed info
    print("\n" + "="*70)
    print(f"TRIPLET #{triplet_idx} DETAILS")
    print("="*70)
    print(f"Anchor File:      {a_path}")
    print(f"Positive File:    {p_path}")
    print(f"Negative File:    {n_path}")
    print()
    print(f"Anchor Person:    {anchor_id}")
    print(f"Positive Person:  {positive_id}  (should match anchor)")
    print(f"Negative Person:  {negative_id}  (should differ)")
    print()
    print(f"Positive Distance: {pos_dist:.6f}  (target: < 0.80)")
    print(f"Negative Distance: {neg_dist:.6f}  (target: >= 0.80)")
    print(f"Margin (d- - d+):  {neg_dist - pos_dist:.6f}")
    print()
    
    if pos_dist > 0.8:
        print(f"FALSE NEGATIVE: Same person rejected (distance {pos_dist:.4f} > 0.80)")
    elif neg_dist < 0.8:
        print(f"FALSE POSITIVE: Different person accepted (distance {neg_dist:.4f} < 0.80)")
    else:
        print(f"CORRECT: Properly classified")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('triplet_idx', type=int, help='Index of the triplet to visualize (0-1010 for test set)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    args = parser.parse_args()
    
    visualize_triplet(args.triplet_idx, args.checkpoint)
