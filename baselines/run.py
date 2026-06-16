# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""CLI entry point for baseline methods.

Usage:
    python -m baselines.run --config-dir <dataset_dir> --method <method_name> [--results-dir DIR] [--debug] [--num-sequences N]
"""

import os
import sys
import argparse
import json
import time
import numpy as np


METHOD_CHOICES = [
    "knn_subseq", "arima", "hawkes", "neural_hawkes", "rmtpp", "online_mlp",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline methods for event prediction")
    parser.add_argument("--config-dir", type=str, required=True,
                        help="Directory containing config.py (dataset configuration)")
    parser.add_argument("--method", type=str, required=True, choices=METHOD_CHOICES,
                        help="Baseline method to run")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory to save predictions and metrics")
    parser.add_argument("--debug", action="store_true",
                        help="Run on 10 sequences only")
    parser.add_argument("--num-sequences", type=int, default=None,
                        help="Number of sequences to run (overrides --debug)")
    parser.add_argument("--hparams", type=str, default=None,
                        help='JSON string of method hyperparameters, e.g. \'{"hidden_size": 64, "epochs": 200}\'')
    return parser.parse_args()


def get_method(method_name, hparams=None):
    """Import and return the method class by name.

    Hyperparameters per method:
        knn_subseq:  window_size (int, default 5), k (int, default 3), type_weight (float, default 0.7)
        arima:       (no tunable hparams — uses internal grid search)
        hawkes:      max_events_per_seq (int, 200), max_history (int, 50), lr (float, 0.05), max_iter (int, 200)
        rmtpp:       hidden_size (int, 32), epochs (int, 100), lr (float, 0.001), patience (int, 10)
        online_mlp:  window_size (int, 3), hidden_size (int, 64), epochs (int, 50), lr (float, 0.001)
    """
    hp = hparams or {}
    if method_name == "knn_subseq":
        from baselines.statistical import KNNSubseq
        return KNNSubseq(**hp)
    elif method_name == "arima":
        from baselines.arima_method import ARIMA
        return ARIMA()
    elif method_name == "hawkes":
        from baselines.hawkes_method import Hawkes
        return Hawkes(**hp)
    elif method_name == "neural_hawkes":
        from baselines.neural_hawkes_method import NeuralHawkes
        return NeuralHawkes(**hp)
    elif method_name == "rmtpp":
        from baselines.rmtpp_method import RMTPP
        return RMTPP(**hp)
    elif method_name == "online_mlp":
        from baselines.mlp_method import OnlineMLP
        return OnlineMLP(**hp)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def build_event_list(record):
    """Convert a dataset record into a list of event dicts."""
    events = []
    for i in range(len(record["type_event"])):
        events.append({
            "type_event": record["type_event"][i],
            "time_since_last_event": record["time_since_last_event"][i],
            "type_text": record["type_text"][i] if record.get("type_text") else "",
        })
    return events


def main():
    args = parse_args()

    # Setup imports: add script dir and config dir to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_dir = os.path.abspath(args.config_dir)

    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

    # Import config
    from config import (
        DATASET_NAME, DATASET_SPLIT, DATASET_CACHE_DIR,
        INIT_HISTORY_RATIO, TIME_UNIT,
    )

    # Import evaluate from project root
    from evaluate import compute_metrics

    # Check for RECORD_CATEGORY_KEY (amazon_review_events)
    config_module = __import__('config')
    record_category_key = getattr(config_module, 'RECORD_CATEGORY_KEY', None)

    # Results directory: baselines/<dataset_name>/results_<method>/
    if args.results_dir:
        results_dir = args.results_dir
    else:
        dataset_name = os.path.basename(config_dir)
        results_dir = os.path.join(script_dir, dataset_name, f"results_{args.method}")
    os.makedirs(results_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}...")
    from datasets import load_dataset
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=DATASET_CACHE_DIR)
    records = list(dataset)

    dataset_filter = getattr(config_module, 'DATASET_FILTER', None)
    if dataset_filter:
        records = [r for r in records if all(r[k] == v for k, v in dataset_filter.items())]
    dataset_max_seq = getattr(config_module, 'DATASET_MAX_SEQUENCES', None)
    if dataset_max_seq and len(records) > dataset_max_seq:
        import random
        random.seed(42)
        records = random.sample(records, dataset_max_seq)

    if args.num_sequences:
        records = records[:args.num_sequences]
    elif args.debug:
        records = records[:10]
    print(f"Loaded {len(records)} sequences")

    # Build event lists and histories
    all_events = []
    histories = []
    init_sizes = []

    for record in records:
        events = build_event_list(record)
        all_events.append(events)
        init_size = int(len(events) * INIT_HISTORY_RATIO)
        init_sizes.append(init_size)
        histories.append(events[:init_size])

    # Initialize method
    hparams = json.loads(args.hparams) if args.hparams else None
    if hparams:
        print(f"Hyperparameters: {hparams}")
    method = get_method(args.method, hparams)

    # Fit on pooled histories (no-op for statistical methods)
    print(f"Fitting {args.method}...")
    t_fit = time.time()
    method.fit(histories)
    print(f"Fit completed in {time.time() - t_fit:.1f}s")

    # Prediction loop
    all_predictions = []
    total_correct = 0
    total_predictions = 0
    t_start = time.time()

    for seq_idx in range(len(records)):
        events = all_events[seq_idx]
        init_size = init_sizes[seq_idx]
        n_steps = len(events) - init_size

        if n_steps <= 0:
            continue

        seq_correct = 0
        for step in range(n_steps):
            target_pos = init_size + step
            true_event = events[target_pos]
            true_type = true_event["type_event"]
            true_time = true_event["time_since_last_event"]

            # Predict
            pred_type, pred_time, description = method.predict(histories[seq_idx])

            prediction = {
                "seq_idx": seq_idx,
                "step": step,
                "target_pos": target_pos,
                "pred_type": pred_type,
                "true_type": true_type,
                "pred_time": float(pred_time),
                "true_time": float(true_time),
                "parse_success": True,
                "generation": description,
            }
            all_predictions.append(prediction)

            if pred_type == true_type:
                seq_correct += 1
            total_predictions += 1

            # Append ground truth to history for next step
            histories[seq_idx].append(events[target_pos])

        total_correct += seq_correct
        seq_acc = seq_correct / n_steps
        elapsed = time.time() - t_start

        if (seq_idx + 1) % max(1, len(records) // 20) == 0 or seq_idx == len(records) - 1:
            print(f"Seq {seq_idx + 1}/{len(records)} | steps={n_steps} | "
                  f"seq_acc={seq_acc:.3f} | total_acc={total_correct / total_predictions:.3f} | "
                  f"elapsed={elapsed:.1f}s")

    # Save predictions
    pred_path = os.path.join(results_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\nSaved {len(all_predictions)} predictions to {pred_path}")

    # Compute metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    compute_metrics(all_predictions, output_path=metrics_path, time_unit=TIME_UNIT)

    print(f"\nTotal time: {time.time() - t_start:.0f}s")
    print(f"Overall accuracy: {total_correct / max(total_predictions, 1):.4f}")


if __name__ == "__main__":
    main()
