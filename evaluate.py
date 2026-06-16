# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_metrics(predictions, output_path=None, time_unit="hours"):
    """Compute all evaluation metrics from a list of prediction dicts.

    Each prediction dict has: pred_type, true_type, pred_time, true_time, step, seq_idx.
    """
    true_types = [p["true_type"] for p in predictions]
    pred_types = [p["pred_type"] for p in predictions]
    true_times = np.array([p["true_time"] for p in predictions])
    pred_times = np.array([p["pred_time"] for p in predictions])

    # --- Type metrics ---
    accuracy = accuracy_score(true_types, pred_types)
    macro_f1 = f1_score(true_types, pred_types, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_types, pred_types, average="weighted", zero_division=0)

    present_labels = sorted(set(true_types + pred_types))
    report = classification_report(
        true_types, pred_types, labels=present_labels, zero_division=0
    )

    # --- Time metrics ---
    abs_errors = np.abs(pred_times - true_times)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((pred_times - true_times) ** 2)))
    median_ae = float(np.median(abs_errors))

    # --- Baselines ---
    from collections import Counter

    type_counts = Counter(true_types)
    most_common_type = type_counts.most_common(1)[0][0]
    baseline_accuracy = type_counts[most_common_type] / len(true_types)
    baseline_mae = float(np.mean(np.abs(true_times - np.mean(true_times))))

    # --- Per-step metrics ---
    step_metrics = defaultdict(lambda: {"true_types": [], "pred_types": [], "true_times": [], "pred_times": []})
    for p in predictions:
        s = p["step"]
        step_metrics[s]["true_types"].append(p["true_type"])
        step_metrics[s]["pred_types"].append(p["pred_type"])
        step_metrics[s]["true_times"].append(p["true_time"])
        step_metrics[s]["pred_times"].append(p["pred_time"])

    per_step = {}
    for step in sorted(step_metrics.keys()):
        sm = step_metrics[step]
        tt = np.array(sm["true_times"])
        pt = np.array(sm["pred_times"])
        per_step[step] = {
            "n": len(sm["true_types"]),
            "accuracy": accuracy_score(sm["true_types"], sm["pred_types"]),
            "mae": float(np.mean(np.abs(pt - tt))),
        }

    # --- Parse failure rate ---
    n_failures = sum(1 for p in predictions if not p.get("parse_success", True))
    parse_fail_rate = n_failures / len(predictions)

    results = {
        "n_predictions": len(predictions),
        "type_accuracy": accuracy,
        "type_macro_f1": macro_f1,
        "type_weighted_f1": weighted_f1,
        f"time_mae_{time_unit}": mae,
        f"time_rmse_{time_unit}": rmse,
        f"time_median_ae_{time_unit}": median_ae,
        "baseline_most_common_type": most_common_type,
        "baseline_accuracy": baseline_accuracy,
        f"baseline_mean_mae_{time_unit}": baseline_mae,
        "parse_failure_rate": parse_fail_rate,
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total predictions: {len(predictions)}")
    print(f"Parse failure rate: {parse_fail_rate:.3f}")
    print()
    print("--- Event Type ---")
    print(f"Accuracy:    {accuracy:.4f}  (baseline: {baseline_accuracy:.4f})")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print()
    print(report)
    print()
    print(f"--- Time ({time_unit}) ---")
    print(f"MAE:       {mae:.4f}  (baseline: {baseline_mae:.4f})")
    print(f"RMSE:      {rmse:.4f}")
    print(f"Median AE: {median_ae:.4f}")
    print()
    print("--- Per-Step Summary (first/last 3) ---")
    sorted_steps = sorted(per_step.keys())
    for step in sorted_steps[:3] + sorted_steps[-3:]:
        ps = per_step[step]
        print(f"  Step {step:3d}: n={ps['n']:4d}, acc={ps['accuracy']:.3f}, mae={ps['mae']:.3f}")
    print("=" * 60)

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"metrics": results, "per_step": {str(k): v for k, v in per_step.items()}}, f, indent=2)
        print(f"Metrics saved to {output_path}")

    return results
