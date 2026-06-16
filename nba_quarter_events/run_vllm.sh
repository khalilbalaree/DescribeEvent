#!/bin/bash
# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Run all experiment variants for nba_quarter_events

set -e

export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"

CONDA_ENV="${CONDA_ENV:-event-prediction}"
CONDA_RUN="conda run --no-capture-output -n $CONDA_ENV"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Determine model short name from config
MODEL_SHORT=$($CONDA_RUN python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from config import MODEL_NAME
print(MODEL_NAME.split('/')[-1] if '/' in MODEL_NAME else MODEL_NAME)
")

# Compute results dir with auto-increment
RESULTS_BASE="$SCRIPT_DIR/results_${MODEL_SHORT}"
RESULTS_DIR="$RESULTS_BASE"
if [ -d "$RESULTS_DIR" ]; then
    i=1
    while [ -d "${RESULTS_BASE}_${i}" ]; do
        i=$((i + 1))
    done
    RESULTS_DIR="${RESULTS_BASE}_${i}"
fi
mkdir -p "$RESULTS_DIR"
exec > >(tee "$RESULTS_DIR/exp_log.txt") 2>&1
echo "Results directory: $RESULTS_DIR"

echo "=== Experiment 1: With type_text ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --results-dir "$RESULTS_DIR" --engine vllm
echo ""; sleep 10

echo "=== Experiment 2: Without type_text ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --results-dir "$RESULTS_DIR" --no-type-text --engine vllm
echo ""; sleep 10

echo "=== Experiment 3: Anonymous types (pure, no text) ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --results-dir "$RESULTS_DIR" --anon-types --engine vllm
echo ""; sleep 10

echo "=== Experiment 4: Anonymous types + type_text ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --results-dir "$RESULTS_DIR" --anon-with-text --engine vllm
echo ""

echo "=== All experiments complete ==="
