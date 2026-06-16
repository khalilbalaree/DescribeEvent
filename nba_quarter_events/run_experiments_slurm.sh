#!/bin/bash
# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#SBATCH --job-name=nba_quarter_vllm
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# Run all experiment variants for nba_quarter_events

set -e

CONDA_ENV="${CONDA_ENV:-event-prediction}"
CONDA_RUN="conda run --no-capture-output -n $CONDA_ENV"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Experiment 1: With type_text ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --engine vllm
echo ""

echo "=== Experiment 2: Without type_text ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --no-type-text --engine vllm
echo ""

echo "=== Experiment 3: Anonymous types (pure, no text) ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --anon-types --engine vllm
echo ""

echo "=== Experiment 4: Anonymous types + type_text ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --anon-with-text --engine vllm
echo ""

echo "=== All experiments complete ==="
