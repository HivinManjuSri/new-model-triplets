"""
analyze_pose_variation.py
-------------------------
Analyze pose variation in your triplet dataset to understand
if angle differences are causing the false negatives.

Usage:
    python analyze_pose_variation.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image

from dataset import TripletFaceDataset
from model import build_model


def estimate_pose_difference(img1, img2):
    """
    Simple heuristic to detect if images have different poses.
    Compares left vs right brightness to detect profile views.
    """
    # Convert to grayscale
    gray1 = img1.convert('L')
    gray2 = img2.convert('L')
    
    # Split into left/right halves
    w1, h1 = gray1.size
    w2, h2 = gray2.size
    
    left1 = np.array(gray1.crop((0, 0, w1//2, h1))).mean()
    right1 = np.array(gray1.crop((w1//2, 0, w1, h1))).mean()
    
    left2 = np.array(gray2.crop((0, 0, w2//2, h2))).mean()
    right2 = np.array(gray2.crop((w2//2, 0, w2, h2))).mean()
    
    # Asymmetry scores (profile views have higher asymmetry)
    asym1 = abs(left1 - right1) / (left1 + right1 + 1e-6)
    asym2 = abs(left2 - right2) / (left2 + right2 + 1e-6)
    
    # If both have similar asymmetry, likely same pose
    # If different asymmetry, likely different poses
    pose_diff = abs(asym1 - asym2)
    
    return pose_diff, asym1, asym2


def analyze_dataset():
    """Analyze pose variation and its effect on distances."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = build_model(pretrained=False).to(device)
    ckpt = torch.load('checkpoints/best_model.pt', map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    
    # Load test dataset
    print("Loading test dataset...")
    dataset = TripletFaceDataset(split='test')
    
    # Analyze triplets
    pose_diffs = []
    pos_distances = []
    neg_distances = []
    false_negatives = []
    
    print("Analyzing triplets...\n")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            # Get tensors
            anchor_t, positive_t, negative_t = dataset[idx]
            
            # Get file paths from dataset
            a_path, p_path, n_path = dataset.triplets[idx]
            
            # Load raw images for pose estimation
            anchor_img = Image.open(a_path).convert('RGB')
            positive_img = Image.open(p_path).convert('RGB')
            negative_img = Image.open(n_path).convert('RGB')
            
            # Estimate pose difference
            pose_diff, asym_a, asym_p = estimate_pose_difference(anchor_img, positive_img)
            
            # Compute embedding distances
            anchor_emb = model(anchor_t.unsqueeze(0).to(device))
            positive_emb = model(positive_t.unsqueeze(0).to(device))
            negative_emb = model(negative_t.unsqueeze(0).to(device))
            
            pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2).item()
            neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2).item()
            
            pose_diffs.append(pose_diff)
            pos_distances.append(pos_dist)
            neg_distances.append(neg_dist)
            
            if pos_dist > 0.8:  # False negative
                false_negatives.append({
                    'idx': idx,
                    'pose_diff': pose_diff,
                    'pos_dist': pos_dist,
                    'asym_anchor': asym_a,
                    'asym_positive': asym_p,
                    'identity': Path(a_path).parent.name
                })
    
    # Convert to numpy
    pose_diffs = np.array(pose_diffs)
    pos_distances = np.array(pos_distances)
    
    # Analyze correlation
    print("="*70)
    print("POSE VARIATION ANALYSIS")
    print("="*70)
    
    # Bin by pose difference
    low_pose = pos_distances[pose_diffs < 0.05]
    med_pose = pos_distances[(pose_diffs >= 0.05) & (pose_diffs < 0.15)]
    high_pose = pos_distances[pose_diffs >= 0.15]
    
    print(f"\nPositive Pair Distance by Pose Variation:")
    print(f"{'Category':<20} {'Count':<10} {'Mean Dist':<12} {'% FN (>0.8)':<12}")
    print("-"*70)
    
    if len(low_pose) > 0:
        fn_rate = (low_pose > 0.8).sum() / len(low_pose) * 100
        print(f"{'Similar Pose':<20} {len(low_pose):<10} {low_pose.mean():<12.4f} {fn_rate:<12.1f}%")
    
    if len(med_pose) > 0:
        fn_rate = (med_pose > 0.8).sum() / len(med_pose) * 100
        print(f"{'Medium Variation':<20} {len(med_pose):<10} {med_pose.mean():<12.4f} {fn_rate:<12.1f}%")
    
    if len(high_pose) > 0:
        fn_rate = (high_pose > 0.8).sum() / len(high_pose) * 100
        print(f"{'Large Variation':<20} {len(high_pose):<10} {high_pose.mean():<12.4f} {fn_rate:<12.1f}%")
    
    # Correlation
    correlation = np.corrcoef(pose_diffs, pos_distances)[0, 1]
    print(f"\nCorrelation (pose diff vs distance): {correlation:.4f}")
    
    if correlation > 0.3:
        print("⚠️  STRONG POSITIVE CORRELATION: Larger pose differences → Larger distances")
        print("    This confirms angle variation is a major issue!")
    elif correlation > 0.15:
        print("⚠️  MODERATE CORRELATION: Pose variation affects distances")
    else:
        print("✓ Weak correlation: Pose variation is not the main issue")
    
    # False negative analysis
    if false_negatives:
        print(f"\n{'='*70}")
        print(f"FALSE NEGATIVES WITH HIGH POSE VARIATION")
        print(f"{'='*70}")
        
        fn_sorted = sorted(false_negatives, key=lambda x: x['pose_diff'], reverse=True)
        
        print(f"{'Triplet':<10} {'Identity':<18} {'Pose Diff':<12} {'Distance':<12} {'Asymmetry A/P':<20}")
        print("-"*70)
        for fn in fn_sorted[:15]:
            print(f"{fn['idx']:<10} {fn['identity']:<18} {fn['pose_diff']:<12.4f} "
                  f"{fn['pos_dist']:<12.4f} {fn['asym_anchor']:.3f} / {fn['asym_positive']:.3f}")
        
        high_pose_fn = [fn for fn in false_negatives if fn['pose_diff'] > 0.15]
        print(f"\nFalse Negatives with high pose variation: {len(high_pose_fn)}/{len(false_negatives)} "
              f"({len(high_pose_fn)/len(false_negatives)*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION:")
    print("="*70)
    
    if correlation > 0.3:
        print("""
Your false negatives are STRONGLY driven by pose variation!

Solutions:
1. Add more pose-varied training data
2. Use pose-augmentation during training
3. Consider 3D face models or pose-invariant architectures
4. Use separate thresholds for frontal vs profile pairs
        """)
    else:
        print("""
Pose variation is not the primary issue.
Other factors (lighting, age, quality) may be more important.
        """)
    
    print("="*70 + "\n")


if __name__ == '__main__':
    analyze_dataset()
