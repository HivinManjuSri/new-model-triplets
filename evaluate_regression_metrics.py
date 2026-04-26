"""
evaluate_regression_metrics.py
------------------------------
Compute regression metrics (MAE, MSE, PMAE) and thresholded F1 score.

This script is useful when your model predicts continuous values and you want:
1) Regression quality metrics:
   - MAE  (Mean Absolute Error)
   - MSE  (Mean Squared Error)
   - PMAE (Percentage Mean Absolute Error)

2) A classification-style F1 score from regression outputs:
   - Convert continuous y_true and y_pred into binary labels using thresholds.
   - Compute precision, recall, and F1 from the resulting confusion counts.

PMAE definition used here:
    PMAE = mean( abs(y_true - y_pred) / max(abs(y_true), eps) ) * 100

Why eps?
- Prevents division-by-zero when y_true contains zeros.

Examples
--------
From CSV (single column each):
    python evaluate_regression_metrics.py --y_true_csv y_true.csv --y_pred_csv y_pred.csv

From CSV with specific column name:
    python evaluate_regression_metrics.py --y_true_csv data.csv --y_pred_csv pred.csv --column value

From inline values:
    python evaluate_regression_metrics.py --y_true "2.0,3.0,4.0,5.0" --y_pred "2.2,2.8,3.9,5.4"

With custom F1 thresholds:
    python evaluate_regression_metrics.py --y_true_csv y_true.csv --y_pred_csv y_pred.csv --positive_threshold 0.7 --prediction_threshold 0.7
"""

import argparse
import csv
import os
from typing import Iterable, List, Tuple


EPS = 1e-12


def parse_inline_values(raw: str) -> List[float]:
    """Parse comma-separated numeric values from CLI input."""
    values: List[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    return values


def read_values_from_csv(path: str, column: str | None = None) -> List[float]:
    """
    Read numeric values from CSV.

    Supported formats:
    - Single-column CSV without header (all rows are parsed as float)
    - Multi-column CSV with header (requires --column)
    """
    values: List[float] = []

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CSV file not found: {path}. "
            "Use an existing path or run with inline values via --y_true and --y_pred."
        )

    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)

        # Try to infer whether a header exists.
        has_header = csv.Sniffer().has_header(sample)

        if has_header:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV appears empty or malformed: {path}")

            # If there are multiple columns and no --column is given, fail clearly.
            if column is None:
                if len(reader.fieldnames) == 1:
                    column = reader.fieldnames[0]
                else:
                    raise ValueError(
                        f"CSV '{path}' has multiple columns {reader.fieldnames}. "
                        "Please provide --column <name>."
                    )

            if column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{column}' not found in {path}. Available: {reader.fieldnames}"
                )

            for row in reader:
                raw = (row.get(column) or "").strip()
                if raw == "":
                    continue
                values.append(float(raw))
        else:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) != 1:
                    raise ValueError(
                        f"CSV '{path}' has multiple columns but no header was detected. "
                        "Add a header and use --column <name>, or provide a single-column CSV."
                    )
                values.append(float(row[0].strip()))

    return values


def validate_lengths(y_true: List[float], y_pred: List[float]) -> None:
    """Ensure inputs are non-empty and aligned."""
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Both y_true and y_pred must contain at least one value.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )


def compute_regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Tuple[float, float, float]:
    """
    Compute MAE, MSE, PMAE.

    Returns:
        (mae, mse, pmae_percent)
    """
    abs_errors: List[float] = []
    sq_errors: List[float] = []
    pct_abs_errors: List[float] = []

    for yt, yp in zip(y_true, y_pred):
        err = yp - yt
        abs_err = abs(err)
        abs_errors.append(abs_err)
        sq_errors.append(err * err)

        # PMAE contribution for one sample.
        # max(abs(yt), EPS) avoids division-by-zero when yt == 0.
        pct_abs_errors.append(abs_err / max(abs(yt), EPS))

    n = len(abs_errors)
    mae = sum(abs_errors) / n
    mse = sum(sq_errors) / n
    pmae = (sum(pct_abs_errors) / n) * 100.0

    return mae, mse, pmae


def compute_f1_from_thresholds(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    positive_threshold: float,
    prediction_threshold: float,
) -> Tuple[float, float, float, int, int, int, int]:
    """
    Compute precision, recall, and F1 by thresholding continuous values.

    Rules:
    - true_label = 1 if y_true >= positive_threshold else 0
    - pred_label = 1 if y_pred >= prediction_threshold else 0
    """
    tp = fp = fn = tn = 0

    for yt, yp in zip(y_true, y_pred):
        true_label = 1 if yt >= positive_threshold else 0
        pred_label = 1 if yp >= prediction_threshold else 0

        if pred_label == 1 and true_label == 1:
            tp += 1
        elif pred_label == 1 and true_label == 0:
            fp += 1
        elif pred_label == 0 and true_label == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)

    return precision, recall, f1, tp, fp, fn, tn


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MAE, MSE, PMAE, and thresholded F1 for regression outputs."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--y_true_csv", type=str, help="CSV path for y_true values")
    input_group.add_argument(
        "--y_true",
        type=str,
        help='Inline y_true values, comma-separated. Example: "1.2,2.3,3.4"',
    )

    parser.add_argument("--y_pred_csv", type=str, help="CSV path for y_pred values")
    parser.add_argument(
        "--y_pred",
        type=str,
        help='Inline y_pred values, comma-separated. Example: "1.0,2.1,3.8"',
    )

    parser.add_argument(
        "--column",
        "--col",
        type=str,
        default=None,
        help="Optional column name when CSV has headers/multiple columns",
    )

    # F1 from regression values requires converting to binary labels.
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=0.5,
        help="Threshold for converting y_true to binary labels (default: 0.5)",
    )
    parser.add_argument(
        "--prediction_threshold",
        type=float,
        default=0.5,
        help="Threshold for converting y_pred to binary labels (default: 0.5)",
    )

    args = parser.parse_args()

    # Keep input mode clear and explicit.
    if args.y_true_csv and not args.y_pred_csv:
        parser.error("When using --y_true_csv, you must also provide --y_pred_csv.")
    if args.y_true and not args.y_pred:
        parser.error("When using --y_true, you must also provide --y_pred.")

    return args


def main() -> None:
    """Load inputs, compute metrics, and print a readable report."""
    args = parse_args()

    try:
        # Load ground truth and predictions from either CSV files or inline lists.
        if args.y_true_csv:
            y_true = read_values_from_csv(args.y_true_csv, args.column)
            y_pred = read_values_from_csv(args.y_pred_csv, args.column)
        else:
            y_true = parse_inline_values(args.y_true)
            y_pred = parse_inline_values(args.y_pred)

        validate_lengths(y_true, y_pred)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Input error: {exc}")
        print("Hint: if you do not have CSV files yet, use:")
        print('  python evaluate_regression_metrics.py --y_true "2,3,4" --y_pred "2.1,2.9,4.2"')
        raise SystemExit(1)

    # Regression metrics.
    mae, mse, pmae = compute_regression_metrics(y_true, y_pred)

    # F1 from thresholded labels.
    precision, recall, f1, tp, fp, fn, tn = compute_f1_from_thresholds(
        y_true,
        y_pred,
        positive_threshold=args.positive_threshold,
        prediction_threshold=args.prediction_threshold,
    )

    print("=" * 70)
    print("REGRESSION METRICS REPORT")
    print("=" * 70)
    print(f"Samples: {len(y_true)}")
    print()

    print("Regression metrics")
    print(f"- MAE  : {mae:.6f}")
    print(f"- MSE  : {mse:.6f}")
    print(f"- PMAE : {pmae:.4f}%")
    print()

    print("Thresholded classification metrics (from regression values)")
    print(f"- positive_threshold   : {args.positive_threshold}")
    print(f"- prediction_threshold : {args.prediction_threshold}")
    print(f"- Precision            : {precision:.6f}")
    print(f"- Recall               : {recall:.6f}")
    print(f"- F1 Score             : {f1:.6f}")
    print()

    print("Confusion counts")
    print(f"- TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print("=" * 70)


if __name__ == "__main__":
    main()
