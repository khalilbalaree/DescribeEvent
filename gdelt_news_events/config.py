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
DATASET_NAME = "XiaoBB/gdelt_news_events"
DATASET_SPLIT = "test"

# Time configuration
TIME_KEY = "time_days"
TIME_UNIT = "days"
TIME_SUFFIX = "d"

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

# Extract event types dynamically from dataset (cached to JSON to avoid
# re-loading inside vLLM worker subprocesses)
import json
_cache_path = os.path.join(os.path.dirname(__file__), ".event_types_cache.json")
if os.path.exists(_cache_path):
    with open(_cache_path) as _f:
        EVENT_TYPES = json.load(_f)
else:
    from datasets import load_dataset
    _dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=DATASET_CACHE_DIR)
    _all_types = set()
    for _row in _dataset:
        for _t in _row["type_event"]:
            _all_types.add(_t)
    EVENT_TYPES = sorted(_all_types)
    with open(_cache_path, "w") as _f:
        json.dump(EVENT_TYPES, _f)

# OpenRouter (used when --engine openrouter)
OPENROUTER_CONFIG = {
    "model": "openai/gpt-5.4-mini",
    "rpm": 60,
    "temperature": 0.7,
    "top_p": None,
    "max_output_tokens": None,
    "reasoning": {"effort": "none"},
    "batch_size": 10,
}

# Gemini — empty = use API defaults
GEMINI_CONFIG = {
    "model": "google/gemini-3-flash-preview",
    "rpm": 60,
    "temperature": None,
    "top_p": None,
    "max_output_tokens": None,
    "reasoning": {"effort": "MINIMAL"},
    "batch_mode": True,
    "batch_poll_interval": 30,
}

# Anonymous type mapping (for ablation)
RANDOM_ANON = False
PREPEND_SEMANTIC_LABEL = True
TYPE_TO_ID = {t: f"type_{i}" for i, t in enumerate(EVENT_TYPES)}

# Shuffled letters (no ordering)
# _ANON_LABELS = "MXKRJWQ"
# TYPE_TO_ID = {t: f"type_{_ANON_LABELS[i]}" for i, t in enumerate(EVENT_TYPES)}

# Non-semantic words
# _ANON_LABELS = ["foo", "bar", "baz", "qux", "wop", "zim", "yak"]
# TYPE_TO_ID = {t: _ANON_LABELS[i] for i, t in enumerate(EVENT_TYPES)}

ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}
