# Event Sequence Prediction Experiments

This repository contains code for event sequence prediction experiments across seven datasets. Each dataset lives in its own directory with a `config.py`, `prompts.py`, and driver scripts. All model experiments use the shared `inference.py` entrypoint at the repository root.

## Datasets

The following dataset configurations are included:

- `amazon_review_events/`
- `earthquake_region_events/`
- `gdelt_news_events/`
- `github_repo_events/`
- `github_user_events/`
- `nba_quarter_events/`
- `wikipedia_edit_events/`

## Prerequisites

Use Python 3.10 or later.

Install the directly imported packages:

```bash
pip install -r requirements.txt
```

For the original pinned vLLM environment, use `requirements-vllm-full.txt`. The full environment file includes transitive dependencies and GPU-specific packages.

Set credentials through environment variables. Do not edit credentials into config files.

```bash
export HF_TOKEN=<your-huggingface-token>        # only needed for gated/private datasets
export OPENROUTER_API_KEY=<your-openrouter-key> # needed for OpenRouter runs
export GEMINI_API_KEY=<your-gemini-key>         # needed for Gemini runs
```

Runtime caches default to `.cache/` inside the repository and can be overridden:

```bash
export EVENT_PREDICTION_CACHE_DIR=/path/to/cache
export MODEL_CACHE_DIR=/path/to/model-cache
export DATASET_CACHE_DIR=/path/to/dataset-cache
```

For local vLLM runs, set GPU visibility in your shell if needed:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

Driver scripts use a Conda environment named `event-prediction` by default. Override it with:

```bash
export CONDA_ENV=<your-conda-env-name>
```

## Running Experiments

Each driver runs four prompt variants:

1. With event type text
2. Without event type text
3. Anonymous event types without text
4. Anonymous event types with text

```bash
# Local vLLM
bash <dataset>/run_experiments.sh

# OpenRouter API
bash <dataset>/run_openrouter.sh

# Gemini API where available
bash gdelt_news_events/run_gemini.sh
```

You can also call the shared entrypoint directly:

```bash
python inference.py --config-dir github_repo_events --engine openrouter --debug
```

Run baseline methods with:

```bash
python -m baselines.run --config-dir github_repo_events --method knn_subseq --debug
```

## Outputs

Experiment outputs are written to dataset-local `results_*` directories. Generated outputs, logs, caches, local environments, and credentials files are ignored by Git.

## Data

This repository does not include or redistribute dataset files. Dataset configurations reference hosted Hugging Face datasets from `DescribeEvents` through `datasets.load_dataset`.

## License

This project is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. See `LICENSE`.
