"""
split_dataset.py
----------------
Scans the triplets_dataset directory, collects all triplet folder names,
shuffles them with a fixed seed for reproducibility, then writes three
split files:
    splits/train.txt
    splits/val.txt
    splits/test.txt

Each line in a split file is the name of one triplet folder
(e.g. triplet_00000).

Run:
    python split_dataset.py
"""

import os
import random
import json

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "triplets_dataset")
SPLITS_DIR  = os.path.join(os.path.dirname(__file__), "splits")

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10   # remainder

SEED = 42
# ────────────────────────────────────────────────────────────────────────────


def main():
    # Collect all triplet folder names
    all_triplets = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.startswith("triplet_")
    ])

    total = len(all_triplets)
    print(f"Found {total} triplet folders.")

    # Shuffle reproducibly
    random.seed(SEED)
    random.shuffle(all_triplets)

    # Compute split indices
    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)
    n_test  = total - n_train - n_val   # absorbs any rounding remainder

    train_set = all_triplets[:n_train]
    val_set   = all_triplets[n_train : n_train + n_val]
    test_set  = all_triplets[n_train + n_val :]

    print(f"Split  ->  train: {len(train_set)}  |  val: {len(val_set)}  |  test: {len(test_set)}")

    # Write split files
    os.makedirs(SPLITS_DIR, exist_ok=True)

    for split_name, split_list in [("train", train_set), ("val", val_set), ("test", test_set)]:
        path = os.path.join(SPLITS_DIR, f"{split_name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(split_list))
        print(f"Wrote {path}")

    # Also write a summary JSON for reference
    summary = {
        "total":       total,
        "train":       len(train_set),
        "val":         len(val_set),
        "test":        len(test_set),
        "seed":        SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio":   VAL_RATIO,
        "test_ratio":  TEST_RATIO,
    }
    summary_path = os.path.join(SPLITS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
