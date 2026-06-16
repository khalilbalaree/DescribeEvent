# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def configure_runtime():
    """Return portable cache paths and configure Hugging Face cache defaults."""
    cache_root = Path(
        os.environ.get("EVENT_PREDICTION_CACHE_DIR", REPO_ROOT / ".cache")
    ).expanduser()
    model_cache_dir = Path(
        os.environ.get("MODEL_CACHE_DIR", cache_root / "models")
    ).expanduser()
    dataset_cache_dir = Path(
        os.environ.get("DATASET_CACHE_DIR", cache_root / "datasets")
    ).expanduser()

    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    return str(model_cache_dir), str(dataset_cache_dir)
