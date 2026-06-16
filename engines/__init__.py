# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    token_count: int
    thinking_token_count: int = 0
    cached_token_count: int = 0


def create_engine(engine_type, config_module, **kwargs):
    """Factory function to create an inference engine.

    Args:
        engine_type: 'vllm' or 'openrouter'
        config_module: the imported config module with model/generation settings
        **kwargs: additional engine-specific arguments (e.g. rpm_override)
    """
    if engine_type == "vllm":
        from engines.vllm_engine import VllmEngine
        return VllmEngine(config_module)
    elif engine_type == "openrouter":
        from engines.openrouter_engine import OpenRouterEngine
        return OpenRouterEngine(config_module, **kwargs)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")
