# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import httpx
from engines import GenerationResult
from engines.base_api_engine import BaseApiEngine, TokenBucketRateLimiter


class OpenRouterEngine(BaseApiEngine):
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config, rpm_override=None):
        config_dict = getattr(config, "OPENROUTER_CONFIG", None) or {}
        super().__init__(config_dict)

        self.api_key = config_dict.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be set in OPENROUTER_CONFIG['api_key'] or OPENROUTER_API_KEY env var"
            )

        self.model = config_dict.get("model", "openai/gpt-oss-120b")
        self.reasoning = config_dict.get("reasoning", None)
        self.provider = config_dict.get("provider", None)
        rpm = rpm_override or config_dict.get("rpm", 20)
        self.rpm = rpm
        self.rate_limiter = TokenBucketRateLimiter(rpm)
        reasoning_desc = f", reasoning={self.reasoning}" if self.reasoning is not None else ""
        provider_desc = f", provider={self.provider}" if self.provider is not None else ""
        print(f"Initialized OpenRouter engine: model={self.model}, rpm={rpm}{reasoning_desc}{provider_desc}")

    def _create_client(self):
        return httpx.AsyncClient(timeout=300.0)

    async def _do_request(self, client, messages):
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if self.provider is not None:
            payload["provider"] = self.provider
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = await client.post(
            self.BASE_URL, json=payload, headers=headers
        )
        if response.status_code == 429:
            error = httpx.HTTPStatusError(
                "Rate limited", request=response.request, response=response
            )
            error.status_code = 429
            error._retry_after = response.headers.get("Retry-After")
            raise error
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        usage = data.get("usage", {})
        token_count = usage.get("completion_tokens", 0)
        cached_token_count = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0
        thinking_token_count = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
        return GenerationResult(
            text=text, token_count=token_count,
            thinking_token_count=thinking_token_count,
            cached_token_count=cached_token_count,
        )

    def _get_retry_after(self, error):
        retry_after = getattr(error, '_retry_after', None)
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
        return None
