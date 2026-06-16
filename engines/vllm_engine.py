# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from vllm import LLM, SamplingParams
from engines import GenerationResult


class VllmEngine:
    def __init__(self, config):
        print("Initializing vLLM...")
        self.model_name = config.MODEL_NAME
        self.llm = LLM(
            model=config.MODEL_NAME,
            tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
            max_model_len=config.MAX_MODEL_LEN,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            download_dir=config.MODEL_CACHE_DIR,
            trust_remote_code=True,
        )
        sampling_kwargs = {"max_tokens": config.MAX_OUTPUT_TOKENS}
        if hasattr(config, "TEMPERATURE"):
            sampling_kwargs["temperature"] = config.TEMPERATURE
        if hasattr(config, "TOP_P"):
            sampling_kwargs["top_p"] = config.TOP_P
        if hasattr(config, "TOP_K"):
            sampling_kwargs["top_k"] = config.TOP_K
        if hasattr(config, "MIN_P"):
            sampling_kwargs["min_p"] = config.MIN_P
        if hasattr(config, "PRESENCE_PENALTY"):
            sampling_kwargs["presence_penalty"] = config.PRESENCE_PENALTY
        if hasattr(config, "REPETITION_PENALTY"):
            sampling_kwargs["repetition_penalty"] = config.REPETITION_PENALTY
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def generate(self, batch_messages):
        chat_kwargs = {}
        if "qwen" in self.model_name.lower():
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        outputs = self.llm.chat(
            batch_messages, self.sampling_params, use_tqdm=False,
            **chat_kwargs,
        )
        return [
            GenerationResult(
                text=output.outputs[0].text,
                token_count=len(output.outputs[0].token_ids),
            )
            for output in outputs
        ]
