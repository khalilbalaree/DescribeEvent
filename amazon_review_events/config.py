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
DATASET_NAME = "XiaoBB/amazon_review_events"
DATASET_SPLIT = "test"

# Time configuration
TIME_KEY = "time_weeks"
TIME_UNIT = "weeks"
TIME_SUFFIX = "w"

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

# Feature flags
INCLUDE_TYPE_TEXT = True

# Per-record category key — signals inference.py to use per-record logic
RECORD_CATEGORY_KEY = "type_category"

# Extract event types per category from dataset
from collections import defaultdict
from datasets import load_dataset

_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=DATASET_CACHE_DIR)
_cat_types = defaultdict(set)
for _row in _dataset:
    for _t in _row["type_event"]:
        _cat_types[_row["type_category"]].add(_t)
CATEGORY_EVENT_TYPES = {cat: sorted(types) for cat, types in _cat_types.items()}

# Per-category anonymous type mappings
# CATEGORY_TYPE_TO_ID = {cat: {t: f"type_{i}" for i, t in enumerate(types)} for cat, types in CATEGORY_EVENT_TYPES.items()}

# Shuffled letters (no ordering)
# _ANON_LABELS = "MXKRJWQZFP"
# CATEGORY_TYPE_TO_ID = {cat: {t: f"type_{_ANON_LABELS[i]}" for i, t in enumerate(types)} for cat, types in CATEGORY_EVENT_TYPES.items()}

# Non-semantic words
# _ANON_LABELS = ["foo", "bar", "baz", "qux", "wop", "zim", "yak", "dex", "piv", "jot"]
# CATEGORY_TYPE_TO_ID = {cat: {t: _ANON_LABELS[i] for i, t in enumerate(types)} for cat, types in CATEGORY_EVENT_TYPES.items()}

# Random anon: each event gets a random type_N label (sanity check)
RANDOM_ANON = False
PREPEND_SEMANTIC_LABEL = True
CATEGORY_TYPE_TO_ID = {cat: {t: f"type_{i}" for i, t in enumerate(types)} for cat, types in CATEGORY_EVENT_TYPES.items()}
CATEGORY_ID_TO_TYPE = {cat: {v: k for k, v in mapping.items()} for cat, mapping in CATEGORY_TYPE_TO_ID.items()}

# OpenRouter (used when --engine openrouter)
OPENROUTER_CONFIG = {
    "model": "openai/gpt-5-mini",
    "rpm": 60,
    "temperature": None,
    "top_p": None,
    "max_output_tokens": None,
    "reasoning": {"effort": "minimal"},
    "batch_size": 10,
}

# Gemini — empty = use API defaults
GEMINI_CONFIG = {
    "model": "gemini-2.5-flash",
    "rpm": 60,
    "temperature": None,
    "top_p": None,
    "max_output_tokens": None,
    "reasoning": {"budget": 0},
    "batch_mode": True,
    "batch_poll_interval": 600,
}

# Backward-compat defaults (inference.py imports these at module level)
# Actual per-record logic overrides these when RECORD_CATEGORY_KEY is set
_first_cat = sorted(CATEGORY_EVENT_TYPES.keys())[0]
EVENT_TYPES = CATEGORY_EVENT_TYPES[_first_cat]
TYPE_TO_ID = CATEGORY_TYPE_TO_ID[_first_cat]
ID_TO_TYPE = CATEGORY_ID_TO_TYPE[_first_cat]
