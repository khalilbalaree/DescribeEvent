# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
import json
import time
import asyncio


def setup_imports(config_dir=None):
    """Set up sys.path so that config.py and prompts.py are found from config_dir,
    and shared modules (parse_output, evaluate) are found from this script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Add script's directory for shared modules (parse_output, evaluate)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Add config_dir for dataset-specific config.py and prompts.py
    if config_dir:
        config_dir = os.path.abspath(config_dir)
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)
    elif script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="GitHub Event Prediction")
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Directory containing config.py and prompts.py")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory to save predictions and metrics")
    parser.add_argument("--debug", action="store_true", help="Run on 10 sequences only")
    parser.add_argument("--num-sequences", type=int, default=None,
                        help="Number of sequences to run (overrides --debug)")
    parser.add_argument("--no-type-text", action="store_true",
                        help="Disable type_text in prompts")
    parser.add_argument("--anon-types", action="store_true",
                        help="Replace type names with anonymous IDs (no text)")
    parser.add_argument("--anon-with-text", action="store_true",
                        help="Anonymous type IDs but keep event descriptions")
    parser.add_argument("--engine", type=str, choices=["vllm", "openrouter"],
                        default="vllm", help="Inference engine (default: vllm)")
    parser.add_argument("--rpm", type=int, default=None,
                        help="Requests per minute for OpenRouter rate limiting")
    parser.add_argument("--cache-friendly", action="store_true", default=None,
                        help="Use sequence-major loop for prompt caching (default: on for openrouter)")
    parser.add_argument("--no-cache-friendly", action="store_true",
                        help="Force step-major loop even for openrouter")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for cache-friendly mode (default: RPM value)")
    return parser.parse_args()


# Parse args early so we can set up imports before any other imports
_args = parse_args()
setup_imports(_args.config_dir)

# Now import config and prompts (from config_dir if specified, else from script dir)
from config import (  # noqa: E402
    MODEL_NAME, TENSOR_PARALLEL_SIZE, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION,
    MODEL_CACHE_DIR, DATASET_NAME, DATASET_SPLIT, DATASET_CACHE_DIR,
    INIT_HISTORY_RATIO, MAX_OUTPUT_TOKENS, EVENT_TYPES,
    ID_TO_TYPE, TIME_KEY, TIME_UNIT, TIME_SUFFIX,
)
from prompts import (  # noqa: E402
    SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_TEXT, SYSTEM_PROMPT_ANON,
    SYSTEM_PROMPT_ANON_WITH_TEXT, build_user_message,
)
# Per-record category support (optional, dataset-specific)
_RECORD_CATEGORY_KEY = getattr(__import__('config'), 'RECORD_CATEGORY_KEY', None)
if _RECORD_CATEGORY_KEY:
    from prompts import get_system_prompt as _get_system_prompt
    from config import CATEGORY_EVENT_TYPES, CATEGORY_ID_TO_TYPE
from parse_output import parse_prediction  # noqa: E402
from evaluate import compute_metrics  # noqa: E402

from datasets import load_dataset  # noqa: E402
from engines import create_engine  # noqa: E402


def build_event_list(record):
    """Convert a dataset record into a list of event dicts."""
    events = []
    for i in range(len(record["type_event"])):
        events.append({
            "type_event": record["type_event"][i],
            "time_since_last_event": record["time_since_last_event"][i],
            "type_text": record["type_text"][i] if record.get("type_text") else "",
        })
    return events


def main():
    args = _args
    # Auto-generate results_dir from model name if not specified
    if args.results_dir:
        results_dir = args.results_dir
    else:
        config_module_tmp = __import__('config')
        if args.engine == "openrouter":
            model_id = (getattr(config_module_tmp, "OPENROUTER_CONFIG", None) or {}).get("model", "google/gemini-2.5-flash")
        else:
            model_id = MODEL_NAME
        # Use part after '/' (e.g. "google/gemini-2.5-flash" -> "gemini-2.5-flash")
        model_short = model_id.split("/")[-1] if "/" in model_id else model_id
        results_dir = os.path.join(args.config_dir or ".", f"results_{model_short}")
        if os.path.exists(results_dir):
            i = 1
            while os.path.exists(f"{results_dir}_{i}"):
                i += 1
            results_dir = f"{results_dir}_{i}"

    anonymize_types = args.anon_types or args.anon_with_text
    include_type_text = args.anon_with_text or (not args.no_type_text and not args.anon_types)
    if args.anon_with_text:
        system_prompt = SYSTEM_PROMPT_ANON_WITH_TEXT
    elif args.anon_types:
        system_prompt = SYSTEM_PROMPT_ANON
    elif include_type_text:
        system_prompt = SYSTEM_PROMPT_WITH_TEXT
    else:
        system_prompt = SYSTEM_PROMPT

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=DATASET_CACHE_DIR)
    records = list(dataset)
    dataset_filter = getattr(__import__('config'), 'DATASET_FILTER', None)
    if dataset_filter:
        records = [r for r in records if all(r[k] == v for k, v in dataset_filter.items())]
    dataset_max_seq = getattr(__import__('config'), 'DATASET_MAX_SEQUENCES', None)
    if dataset_max_seq and len(records) > dataset_max_seq:
        import random
        random.seed(42)
        records = random.sample(records, dataset_max_seq)
    if args.num_sequences:
        records = records[:args.num_sequences]
    elif args.debug:
        records = records[:10]
    print(f"Loaded {len(records)} sequences")

    # Initialize inference engine
    config_module = __import__('config')
    engine_kwargs = {}
    if args.engine != "vllm" and args.rpm is not None:
        engine_kwargs["rpm_override"] = args.rpm
    engine = create_engine(args.engine, config_module, **engine_kwargs)

    # Build event lists and initialize histories
    all_events = []
    histories = []
    init_sizes = []

    for record in records:
        events = build_event_list(record)
        all_events.append(events)
        init_size = int(len(events) * INIT_HISTORY_RATIO)
        init_sizes.append(init_size)
        histories.append(events[:init_size])

    # Compute median time for fallback
    all_times = []
    for events in all_events:
        for e in events:
            all_times.append(e["time_since_last_event"])
    median_time = float(sorted(all_times)[len(all_times) // 2])
    print(f"Median time between events: {median_time:.3f}{TIME_SUFFIX}")

    # Determine max steps needed
    max_remaining = max(len(events) - init_sizes[i] for i, events in enumerate(all_events))
    print(f"Max remaining steps: {max_remaining}")

    all_predictions = []
    total_parse_failures = 0
    total_predictions = 0
    t_start = time.time()

    # Determine output paths
    if args.anon_with_text:
        suffix = "_anon_text"
    elif args.anon_types:
        suffix = "_anon"
    elif args.no_type_text:
        suffix = "_no_text"
    else:
        suffix = ""
    pred_filename = f"predictions{suffix}.json"
    metrics_filename = f"metrics{suffix}.json"
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        pred_path = os.path.join(results_dir, pred_filename)
        metrics_path = os.path.join(results_dir, metrics_filename)
    else:
        pred_path = pred_filename
        metrics_path = metrics_filename

    # Incremental save file (JSONL for append-friendliness)
    jsonl_path = pred_path + "l"  # e.g. predictions.jsonl

    completed_seq_indices = set()

    # Determine loop order: cache-friendly (sequence-major) vs step-major
    cache_friendly = args.cache_friendly
    if args.no_cache_friendly:
        cache_friendly = False
    elif cache_friendly is None:
        cache_friendly = (args.engine != "vllm")

    def _build_messages(seq_idx, category, history=None):
        """Build the prompt messages for a given sequence.

        If history is provided, use it instead of histories[seq_idx].
        """
        user_msg_kwargs = dict(
            include_type_text=include_type_text,
            anonymize_types=anonymize_types,
        )
        if category:
            user_msg_kwargs["category"] = category
        user_msg = build_user_message(
            records[seq_idx]["description"],
            history if history is not None else histories[seq_idx],
            **user_msg_kwargs,
        )
        if _RECORD_CATEGORY_KEY:
            seq_system_prompt = _get_system_prompt(category, include_type_text=include_type_text, anonymize_types=anonymize_types)
        else:
            seq_system_prompt = system_prompt
        return [
            {"role": "system", "content": seq_system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def _parse_and_record(seq_idx, step, target_pos, category, result):
        """Parse a model result and record the prediction. Returns (prediction, parse_ok)."""
        response_text = result.text
        if _RECORD_CATEGORY_KEY and category:
            rec_event_types = CATEGORY_EVENT_TYPES[category]
            rec_anon_to_type = CATEGORY_ID_TO_TYPE[category] if anonymize_types else None
        else:
            rec_event_types = EVENT_TYPES
            rec_anon_to_type = ID_TO_TYPE if anonymize_types else None
        pred_type, pred_time, parse_ok = parse_prediction(
            response_text, event_types=rec_event_types,
            fallback_time=median_time,
            anon_to_type=rec_anon_to_type,
            time_key=TIME_KEY,
        )
        n_out_tokens = result.token_count
        n_thinking_tokens = result.thinking_token_count
        n_cached_tokens = result.cached_token_count
        true_event = all_events[seq_idx][target_pos]
        true_type = true_event["type_event"]
        true_time = true_event["time_since_last_event"]
        extra_str = ""
        if n_thinking_tokens:
            extra_str += f" | think_tokens={n_thinking_tokens}"
        if n_cached_tokens:
            extra_str += f" | cached_tokens={n_cached_tokens}"
        print(f"  seq={seq_idx} pos={target_pos} | pred={pred_type}/{pred_time:.3f}{TIME_SUFFIX} | "
              f"true={true_type}/{true_time:.3f}{TIME_SUFFIX} | "
              f"parsed={parse_ok} | out_tokens={n_out_tokens}{extra_str}", flush=True)
        if not parse_ok:
            print(f"    RAW OUTPUT: {response_text[:300]}", flush=True)
        prediction = {
            "seq_idx": seq_idx,
            "step": step,
            "target_pos": target_pos,
            "pred_type": pred_type,
            "true_type": true_type,
            "pred_time": pred_time,
            "true_time": true_time,
            "parse_success": parse_ok,
            "generation": response_text,
        }
        if n_thinking_tokens:
            prediction["thinking_token_count"] = n_thinking_tokens
        if n_cached_tokens:
            prediction["cached_token_count"] = n_cached_tokens
        return prediction, parse_ok

    if getattr(engine, 'batch_mode', False):
        # Batch mode: pre-build all prompts and submit as one batch job
        print("\nBatch mode: pre-building all prompts...")
        all_batch_messages = []
        all_batch_meta = []
        for seq_idx in range(len(records)):
            category = records[seq_idx][_RECORD_CATEGORY_KEY] if _RECORD_CATEGORY_KEY else None
            history = list(histories[seq_idx])  # copy initial history
            n_steps = len(all_events[seq_idx]) - init_sizes[seq_idx]
            for step in range(n_steps):
                target_pos = init_sizes[seq_idx] + step
                messages = _build_messages(seq_idx, category, history=list(history))
                all_batch_messages.append(messages)
                all_batch_meta.append((seq_idx, step, target_pos, category))
                # Append ground truth for next step's history
                history.append(all_events[seq_idx][target_pos])

        print(f"Submitting {len(all_batch_messages)} prompts as batch...")
        results = engine.generate(all_batch_messages)

        for i, result in enumerate(results):
            seq_idx, step, target_pos, category = all_batch_meta[i]
            prediction, parse_ok = _parse_and_record(seq_idx, step, target_pos, category, result)
            all_predictions.append(prediction)
            if not parse_ok:
                total_parse_failures += 1
        total_predictions = len(all_batch_meta)

    elif cache_friendly:
        # Dynamic async worker mode: N workers each process one sequence at a time
        # (all steps), then immediately grab the next sequence from the pool.
        # Requests fire as soon as a rate-limit slot opens — no waiting for a
        # whole batch to finish before starting new work.
        default_rpm = getattr(engine, 'rpm', 60)
        engine_batch_size = getattr(engine, 'batch_size', None)
        num_workers = args.batch_size or engine_batch_size or default_rpm
        print(f"\nUsing dynamic cache-friendly mode ({num_workers} workers)")

        # Build pool of sequence indices that need processing
        seq_pool = []
        for seq_idx in range(len(records)):
            if seq_idx in completed_seq_indices:
                continue
            n_steps = len(all_events[seq_idx]) - init_sizes[seq_idx]
            if n_steps > 0:
                seq_pool.append(seq_idx)
        total_seqs = len(seq_pool) + len(completed_seq_indices)

        # Open JSONL file for incremental writes (append mode)
        jsonl_file = open(jsonl_path, "w")

        # Thread-safe state for workers (asyncio is single-threaded, but
        # we protect shared mutation with a lock for clarity)
        pool_lock = asyncio.Lock()
        results_lock = asyncio.Lock()
        seqs_completed = len(completed_seq_indices)

        async def worker(client):
            nonlocal seqs_completed, total_parse_failures, total_predictions

            while True:
                # Grab next sequence from pool
                async with pool_lock:
                    if not seq_pool:
                        return
                    seq_idx = seq_pool.pop(0)

                category = records[seq_idx][_RECORD_CATEGORY_KEY] if _RECORD_CATEGORY_KEY else None
                n_steps = len(all_events[seq_idx]) - init_sizes[seq_idx]
                seq_correct = 0
                seq_failures = 0
                seq_predictions = []
                t_seq = time.time()

                for step in range(n_steps):
                    target_pos = init_sizes[seq_idx] + step
                    messages = _build_messages(seq_idx, category)
                    result = await engine.generate_one(client, messages)
                    prediction, parse_ok = _parse_and_record(
                        seq_idx, step, target_pos, category, result)

                    seq_predictions.append(prediction)
                    if not parse_ok:
                        seq_failures += 1
                    if prediction["pred_type"] == prediction["true_type"]:
                        seq_correct += 1

                    # Append ground truth to history for next step
                    histories[seq_idx].append(all_events[seq_idx][target_pos])

                # Sequence finished — flush predictions to JSONL and update state
                async with results_lock:
                    all_predictions.extend(seq_predictions)
                    total_predictions += len(seq_predictions)
                    total_parse_failures += seq_failures
                    # Write to JSONL
                    for pred in seq_predictions:
                        jsonl_file.write(json.dumps(pred) + "\n")
                    jsonl_file.flush()

                    seqs_completed += 1
                    seq_elapsed = time.time() - t_seq
                    seq_acc = seq_correct / n_steps
                    elapsed_total = time.time() - t_start
                    print(
                        f"Seq {seq_idx} done | {seqs_completed}/{total_seqs} | "
                        f"steps={n_steps} | time={seq_elapsed:.1f}s | "
                        f"total={elapsed_total:.0f}s | acc={seq_acc:.3f} | "
                        f"parse_fail={seq_failures}/{n_steps}"
                    )

        if seq_pool:
            engine.run_workers(worker, min(num_workers, len(seq_pool)))
        jsonl_file.close()
    else:
        # Step-major loop: process all sequences at the same step before moving on.
        # Better for vLLM which benefits from large batch parallelism.
        print(f"\nUsing step-major loop order")
        for step in range(max_remaining):
            batch_messages = []
            batch_meta = []

            for seq_idx in range(len(records)):
                target_pos = init_sizes[seq_idx] + step
                if target_pos >= len(all_events[seq_idx]):
                    continue
                category = records[seq_idx][_RECORD_CATEGORY_KEY] if _RECORD_CATEGORY_KEY else None
                messages = _build_messages(seq_idx, category)
                batch_messages.append(messages)
                batch_meta.append((seq_idx, target_pos, category))

            if not batch_messages:
                break

            print(f"\n>>> Step {step}: sending {len(batch_messages)} prompts to model...", flush=True)
            t_step = time.time()
            results = engine.generate(batch_messages)
            step_elapsed = time.time() - t_step
            print(f">>> Step {step}: inference done in {step_elapsed:.1f}s", flush=True)

            step_failures = 0
            for i, result in enumerate(results):
                seq_idx, target_pos, category = batch_meta[i]
                prediction, parse_ok = _parse_and_record(
                    seq_idx, step, target_pos, category, result)
                all_predictions.append(prediction)
                if not parse_ok:
                    step_failures += 1

            total_parse_failures += step_failures
            total_predictions += len(batch_meta)

            # Append ground truth to histories for next step
            for seq_idx, target_pos, _cat in batch_meta:
                histories[seq_idx].append(all_events[seq_idx][target_pos])

            elapsed_total = time.time() - t_start
            step_acc = sum(
                1 for p in all_predictions[-len(batch_meta):]
                if p["pred_type"] == p["true_type"]
            ) / len(batch_meta)

            print(
                f"Step {step:3d}/{max_remaining} | batch={len(batch_meta):4d} | "
                f"step_time={step_elapsed:.1f}s | total={elapsed_total:.0f}s | "
                f"step_acc={step_acc:.3f} | parse_fail={step_failures}/{len(batch_meta)}"
            )

    # Save predictions (final JSON format)
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    # Remove JSONL temp file now that final JSON is written
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    print(f"\nSaved {len(all_predictions)} predictions to {pred_path}")

    # Compute metrics
    compute_metrics(all_predictions, output_path=metrics_path, time_unit=TIME_UNIT)

    print(f"\nTotal time: {time.time() - t_start:.0f}s")
    print(f"Parse failure rate: {total_parse_failures}/{total_predictions} = {total_parse_failures/max(total_predictions,1):.3f}")


if __name__ == "__main__":
    main()
