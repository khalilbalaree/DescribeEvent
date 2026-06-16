# Event Sequence Prediction Experiments

Code for event sequence prediction experiments across seven datasets.

## Datasets

- `amazon_review_events/`
- `earthquake_region_events/`
- `gdelt_news_events/`
- `github_repo_events/`
- `github_user_events/`
- `nba_quarter_events/`
- `wikipedia_edit_events/`

## Prerequisites

Python 3.10+.

```bash
pip install -r requirements.txt
```

```bash
export OPENROUTER_API_KEY=<your-openrouter-key>
```

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
# OpenRouter API
bash <dataset>/run_openrouter.sh

# Local vLLM
bash <dataset>/run_vllm.sh
```

```bash
python inference.py --config-dir github_repo_events --engine openrouter --debug
```

```bash
python -m baselines.run --config-dir github_repo_events --method knn_subseq --debug
```

## Outputs

Experiment outputs are written to dataset-local `results_*` directories.

## Data

Dataset configurations use public Hugging Face datasets from `DescribeEvents`.

## License

This project is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. See `LICENSE`.
