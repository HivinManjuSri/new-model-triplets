# Final Protocol Notes for Paper

This document defines one consistent set of facts for writing the paper.

## 1) Embedding Dimension (Final)

Final model embedding size is **128**.

- Code definition uses `EMBEDDING_DIM = 128` in model implementation.
- The embedding head projects to the configured embedding dimension, and current config is 128.
- Any earlier 512-d mention should be treated as prior experimentation, not the final reported architecture.

## 2) Dataset Description (Use This Version)

### 2.1 Raw vs cleaned data

- **Raw images:** original pre-cleaning count is not tracked in the current repository metadata.
- **Cleaned images used in this project:** **131,405** total images.

Image counts by split (folder counts):

- Train images: **105,997**
- Validation images: **12,579**
- Test images: **12,829**
- Total images: **131,405**

### 2.2 Training unit

The model is trained with **triplets** (Anchor, Positive, Negative), not single-image classification labels.

Triplet counts from CSV files (excluding headers):

- Train triplets: **86,400**
- Validation triplets: **10,800**
- Test triplets: **10,800**
- Total triplets: **108,000**

So, for paper wording:

- Data source is image-level face data.
- Optimization unit is triplets.
- Evaluation is pairwise verification from triplet-derived genuine/impostor pairs.

## 3) Split and Evaluation Protocol (Final)

Identity sets are disjoint across splits:

- Train identities: **432**
- Validation identities: **54**
- Test identities: **54**
- Overlap train-val: **0**
- Overlap train-test: **0**
- Overlap val-test: **0**

This is an **open-set verification** protocol (identity-disjoint).

## 4) What to Report in Paper

Primary metrics (recommended for open-set verification):

- ROC-AUC
- FAR
- FRR
- EER
- TAR at fixed FAR (for example TAR @ FAR = 0.1%)

Optional secondary metrics (only with clear wording):

- Accuracy
- Precision
- Recall
- Specificity
- Confusion matrix

Important wording constraint:

- If reporting accuracy/precision/recall/specificity, describe them as **threshold-based verification decision metrics on genuine vs impostor pairs**, not closed-set multiclass identity-classification metrics.

## 5) Suggested Paper Text

"We train a Siamese triplet-loss face verification model with a 128-dimensional L2-normalized embedding. The cleaned dataset contains 131,405 face images, from which 108,000 triplets are formed (86,400 train, 10,800 validation, 10,800 test). Splits are identity-disjoint (open-set protocol), and evaluation is performed as face verification using ROC-AUC, FAR/FRR, and EER, with additional threshold-based pairwise metrics reported for completeness."
