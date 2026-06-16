# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import re
import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from engines import GenerationResult
from engines.base_api_engine import BaseApiEngine, TokenBucketRateLimiter


class GeminiEngine(BaseApiEngine):

    def __init__(self, config, rpm_override=None):
        config_dict = getattr(config, "GEMINI_CONFIG", None) or {}
        super().__init__(config_dict)

        api_key = config_dict.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key must be set in GEMINI_CONFIG['api_key'] or GEMINI_API_KEY env var"
            )

        self.model = config_dict.get("model", "gemini-3-flash-preview")
        rpm = rpm_override or config_dict.get("rpm", 60)
        self.rpm = rpm
        self.rate_limiter = TokenBucketRateLimiter(rpm)
        self.api_key = api_key

        # Batch mode config
        self.batch_mode = config_dict.get("batch_mode", False)
        self.batch_poll_interval = config_dict.get("batch_poll_interval", 30)

        # Reasoning config: dict with optional "effort" (MINIMAL/LOW/MEDIUM/HIGH) and/or "budget" (int tokens, 0 to disable)
        self.reasoning = config_dict.get("reasoning", None)
        if self.reasoning is not None:
            thinking_kwargs = {}
            if "budget" in self.reasoning:
                thinking_kwargs["thinking_budget"] = self.reasoning["budget"]
            if "effort" in self.reasoning:
                level_map = {"LOW": types.ThinkingLevel.LOW, "MEDIUM": types.ThinkingLevel.MEDIUM, "HIGH": types.ThinkingLevel.HIGH, "MINIMAL": types.ThinkingLevel.MINIMAL}
                thinking_kwargs["thinking_level"] = level_map[self.reasoning["effort"].upper()]
            self.thinking_config = types.ThinkingConfig(**thinking_kwargs) if thinking_kwargs else None
        else:
            self.thinking_config = None

        reasoning_desc = f", reasoning={self.reasoning}" if self.reasoning is not None else ""
        batch_desc = ", batch_mode=True" if self.batch_mode else ""
        print(f"Initialized Gemini engine: model={self.model}, rpm={rpm}{reasoning_desc}{batch_desc}")

    @property
    def _batch_model_name(self):
        """Model name for batch API — strip 'google/' prefix if present."""
        model = self.model
        if model.startswith("google/"):
            model = model[len("google/"):]
        return model

    def _build_gen_config_kwargs(self, system_instruction=None):
        """Build shared generation config kwargs used by both async and batch paths."""
        gen_kwargs = {}
        if system_instruction is not None:
            gen_kwargs["system_instruction"] = system_instruction
        if self.temperature is not None:
            gen_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            gen_kwargs["max_output_tokens"] = self.max_tokens
        if self.thinking_config is not None:
            gen_kwargs["thinking_config"] = self.thinking_config
        return gen_kwargs

    def _create_client(self):
        return genai.Client(api_key=self.api_key)

    def generate(self, batch_messages):
        if self.batch_mode:
            return self._generate_batch_api(batch_messages)
        return super().generate(batch_messages)

    def _generate_batch_api(self, batch_messages):
        """Submit all requests as a single Gemini Batch API job."""
        # Build inline requests
        inline_requests = []
        for messages in batch_messages:
            system_instruction = None
            user_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]

            gen_kwargs = self._build_gen_config_kwargs(system_instruction)
            request = types.InlinedRequest(
                model=self._batch_model_name,
                contents=user_content,
                config=types.GenerateContentConfig(**gen_kwargs),
            )
            inline_requests.append(request)

        print(f"Submitting batch job with {len(inline_requests)} requests...")
        client = genai.Client(api_key=self.api_key)
        batch_job = client.batches.create(
            model=self._batch_model_name,
            src=inline_requests,
            config=types.CreateBatchJobConfig(display_name="event-prediction"),
        )
        print(f"Batch job created: {batch_job.name} (state={batch_job.state})")

        # Poll until terminal state
        terminal_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }
        total_requests = len(batch_messages)
        while True:
            state = str(batch_job.state) if batch_job.state else ""
            # Handle both enum name and raw string
            state_name = state.split(".")[-1] if "." in state else state
            if state_name in terminal_states:
                break
            # Print progress counters from completion_stats if available
            stats = batch_job.completion_stats
            succeeded = (stats.successful_count or 0) if stats else 0
            failed = (stats.failed_count or 0) if stats else 0
            completed = succeeded + failed
            progress = f" — {completed}/{total_requests} done ({succeeded} ok, {failed} failed)" if completed else ""
            print(f"  Batch status: {state_name}{progress} — polling in {self.batch_poll_interval}s...")
            time.sleep(self.batch_poll_interval)
            batch_job = client.batches.get(name=batch_job.name)

        state_name = str(batch_job.state).split(".")[-1] if "." in str(batch_job.state) else str(batch_job.state)
        stats = batch_job.completion_stats
        succeeded = (stats.successful_count or 0) if stats else 0
        failed = (stats.failed_count or 0) if stats else 0
        print(f"Batch job finished: {state_name} ({succeeded} succeeded, {failed} failed)")

        if state_name != "JOB_STATE_SUCCEEDED":
            print(f"Batch job did not succeed (state={state_name}), returning empty results")
            return [GenerationResult(text="", token_count=0)] * len(batch_messages)

        # Parse results from inlined responses
        results = []
        responses = batch_job.dest.inlined_responses
        for resp in responses:
            response = resp.response
            if response and response.candidates:
                text = response.text or ""
                token_count = 0
                thinking_token_count = 0
                cached_token_count = 0
                if response.usage_metadata:
                    token_count = response.usage_metadata.candidates_token_count or 0
                    thinking_token_count = getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0
                    cached_token_count = getattr(response.usage_metadata, 'cached_content_token_count', 0) or 0
                results.append(GenerationResult(
                    text=text, token_count=token_count,
                    thinking_token_count=thinking_token_count,
                    cached_token_count=cached_token_count,
                ))
            else:
                # Failed individual request
                error = getattr(resp, 'error', None)
                if error:
                    print(f"  Batch request error: {error}")
                results.append(GenerationResult(text="", token_count=0))

        return results

    async def _do_request(self, client, messages):
        # Extract system instruction from messages
        system_instruction = None
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]

        gen_kwargs = self._build_gen_config_kwargs(system_instruction)
        config = types.GenerateContentConfig(**gen_kwargs)

        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=user_content,
                config=config,
            )
        except genai_errors.APIError as e:
            if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
                # Re-raise with status_code so base class retry logic recognizes it
                e.status_code = 429
                # Try to extract retry delay from error message
                match = re.search(r'retry in ([\d.]+)s', str(e))
                if match:
                    e._retry_after = float(match.group(1))
                raise
            raise

        text = response.text or ""
        token_count = 0
        thinking_token_count = 0
        cached_token_count = 0
        if response.usage_metadata:
            token_count = response.usage_metadata.candidates_token_count or 0
            thinking_token_count = getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0
            cached_token_count = getattr(response.usage_metadata, 'cached_content_token_count', 0) or 0

        return GenerationResult(
            text=text, token_count=token_count,
            thinking_token_count=thinking_token_count,
            cached_token_count=cached_token_count,
        )

    def _get_retry_after(self, error):
        retry_after = getattr(error, '_retry_after', None)
        if retry_after:
            return retry_after
        return None

    async def _cleanup_client(self, client):
        # genai.Client doesn't need explicit cleanup
        pass
