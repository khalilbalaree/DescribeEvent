#!/bin/bash
# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Run all baseline methods for a given dataset.
# Usage: bash baselines/run_baselines.sh <config_dir> [extra_args...]
#
# Example:
#   bash baselines/run_baselines.sh github_repo_events --debug
#   bash baselines/run_baselines.sh amazon_review_events
#
# Per-method hyperparameters can be customized below.

set -e

CONFIG_DIR="${1:?Usage: $0 <config_dir> [extra_args...]}"
shift
EXTRA_ARGS="$@"

cd "$(dirname "$0")/.."

# ── Hyperparameters per method (edit as needed) ──────────────────────
declare -A HPARAMS
HPARAMS[knn_subseq]='{"window_size": 3, "k": 3, "type_weight": 0.5}'
HPARAMS[arima]=""
HPARAMS[hawkes]='{"max_events_per_seq": 200, "max_history": 50, "lr": 0.01, "max_iter": 100}'
HPARAMS[neural_hawkes]='{"hidden_size": 64, "epochs": 100, "lr": 0.01, "patience": 10}'
HPARAMS[rmtpp]='{"hidden_size": 64, "epochs": 100, "lr": 0.01, "patience": 10}'
HPARAMS[online_mlp]='{"window_size": 3, "hidden_size": 64, "epochs": 100, "lr": 0.01, "patience": 10}'
# ─────────────────────────────────────────────────────────────────────

METHODS=(knn_subseq arima hawkes neural_hawkes rmtpp online_mlp)

for method in "${METHODS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running: $method on $CONFIG_DIR"
    echo "========================================"

    hp="${HPARAMS[$method]}"
    if [ -n "$hp" ]; then
        echo "  hparams: $hp"
        python -m baselines.run --config-dir "$CONFIG_DIR" --method "$method" --hparams "$hp" $EXTRA_ARGS
    else
        python -m baselines.run --config-dir "$CONFIG_DIR" --method "$method" $EXTRA_ARGS
    fi

    echo ""
done

echo "All baselines completed for $CONFIG_DIR"
