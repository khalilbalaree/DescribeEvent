#!/bin/bash
# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Run all experiment variants for gdelt_news_events using Gemini engine (batch mode)

set -e

export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before running this script.}"

CONDA_ENV="${CONDA_ENV:-event-prediction}"
CONDA_RUN="conda run --no-capture-output -n $CONDA_ENV"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Determine model short name from config
MODEL_SHORT=$($CONDA_RUN python -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from config import GEMINI_CONFIG
m = GEMINI_CONFIG.get('model', 'gemini-3-flash-preview')
print(m.split('/')[-1] if '/' in m else m)
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
LOG_FILE="$RESULTS_DIR/exp_log.txt"
echo "Results directory: $RESULTS_DIR"
echo "Logging to: $LOG_FILE"

# Redirect all output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Experiment 1: With type_text ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --engine gemini --results-dir "$RESULTS_DIR"
echo ""; sleep 10

echo "=== Experiment 2: Without type_text ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --engine gemini --results-dir "$RESULTS_DIR" --no-type-text
echo ""; sleep 10

echo "=== Experiment 3: Anonymous types (pure, no text) ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --engine gemini --results-dir "$RESULTS_DIR" --anon-types
echo ""; sleep 10

echo "=== Experiment 4: Anonymous types + type_text ==="
$CONDA_RUN python "$PARENT_DIR/inference.py" --config-dir "$SCRIPT_DIR" --engine gemini --results-dir "$RESULTS_DIR" --anon-with-text
echo ""

echo "=== All experiments complete ==="
