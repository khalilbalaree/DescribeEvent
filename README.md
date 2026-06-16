# Can LLMs Access their World Knowledge for Event Prediction?

Recent work has questioned whether LLMs can use pretrained world knowledge for temporal event prediction. This project studies event forecasting as an interface problem: grounding events with natural-language descriptions and using symbolic labels to separate event semantics from the prediction space. It introduces DESCRIBEEVENT, a benchmark of about 150K events across seven datasets and six domains, and evaluates when capable LLMs benefit from descriptions and symbolic labels compared with description-free variants and non-LLM baselines.

## Datasets

Hugging Face: https://huggingface.co/DescribeEvents

Each folder is an experiment entry for a different dataset.

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

## Running Experiments

Each driver runs four prompt variants:

1. Semantic event type with description
2. Semantic event type without description
3. Symbolic event type without description
4. Symbolic event type with description

```bash
# OpenRouter API
bash <dataset>/run_openrouter.sh
```

Local vLLM experiments use `TENSOR_PARALLEL_SIZE = 2` by default and were run on 2 NVIDIA RTX A6000 48GB GPUs. Set `CUDA_VISIBLE_DEVICES=0,1` to choose the two GPUs.

```bash
# Local vLLM
CUDA_VISIBLE_DEVICES=0,1 bash <dataset>/run_vllm.sh
```

## Outputs

Experiment outputs are written to dataset-local `results_*` directories.

## Data

Dataset configurations use public Hugging Face datasets from `DescribeEvents`.

## License

This project is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. See `LICENSE`.
