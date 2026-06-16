# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from runtime_config import configure_runtime

MODEL_CACHE_DIR, DATASET_CACHE_DIR = configure_runtime()

# Model
MODEL_NAME = "Qwen/Qwen3.5-27B"
TENSOR_PARALLEL_SIZE = 2
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.90

# Dataset
DATASET_NAME = "DescribeEvents/github_user_events"
DATASET_SPLIT = "test"

# Evaluation
INIT_HISTORY_RATIO = 0.9

# Generation
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 1.5
REPETITION_PENALTY = 1.0
# TEMPERATURE = 0.6
# TOP_P = 0.9
MAX_OUTPUT_TOKENS = 4096

# Time configuration
TIME_KEY = "time_hours"
TIME_UNIT = "hours"
TIME_SUFFIX = "h"

# Feature flags
INCLUDE_TYPE_TEXT = True

# Canonical event types
EVENT_TYPES = [
    "push",
    "comment",
    "pr_merged",
    "pr_reviewed",
    "pr_opened",
    "issue_closed",
    "issue_opened",
    "release",
]

# OpenRouter (used when --engine openrouter)
OPENROUTER_CONFIG = {
    "model": "google/gemini-2.5-flash",
    "rpm": 60,
    "temperature": None,
    "top_p": None,
    "max_output_tokens": None,
    "reasoning": {"effort": "none"},
    "batch_size": 10,
}

# Anonymous type mapping (for ablation)
# TYPE_TO_ID = {t: f"type_{i}" for i, t in enumerate(EVENT_TYPES)}

# Shuffled letters (no ordering)
# _ANON_LABELS = "MXKRJWQZ"
# TYPE_TO_ID = {t: f"type_{_ANON_LABELS[i]}" for i, t in enumerate(EVENT_TYPES)}

# Non-semantic words
# _ANON_LABELS = ["foo", "bar", "baz", "qux", "wop", "zim", "yak", "dex"]
# TYPE_TO_ID = {t: _ANON_LABELS[i] for i, t in enumerate(EVENT_TYPES)}

# Random anon: each event gets a random type_N label (sanity check)
RANDOM_ANON = False
PREPEND_SEMANTIC_LABEL = True
TYPE_TO_ID = {t: f"type_{i}" for i, t in enumerate(EVENT_TYPES)}

ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}
