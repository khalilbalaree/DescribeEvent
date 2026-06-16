# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import asyncio
from engines import GenerationResult


class TokenBucketRateLimiter:
    def __init__(self, rpm, burst=None):
        self.rate = rpm / 60.0        # tokens per second
        self.burst = burst or rpm     # max bucket size
        self.tokens = float(self.burst)  # start full → immediate burst
        self.last_refill = 0.0
        self._lock = None

    async def acquire(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = asyncio.get_event_loop().time()
            # Refill tokens based on elapsed time
            if self.last_refill > 0:
                self.tokens = min(self.burst, self.tokens + (now - self.last_refill) * self.rate)
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            # Wait for 1 token
            wait = (1.0 - self.tokens) / self.rate
            await asyncio.sleep(wait)
            self.last_refill = asyncio.get_event_loop().time()
            self.tokens = 0.0

    def drain(self):
        """Called on 429 — force all workers to wait for refill."""
        self.tokens = 0.0


class BaseApiEngine:
    """Shared base class for API-based inference engines.

    Subclasses must implement:
        _create_client() -> client object for async requests
        _do_request(client, messages) -> GenerationResult
    """
    MAX_RETRIES = 5

    def __init__(self, config_dict):
        self.temperature = config_dict.get("temperature")
        self.top_p = config_dict.get("top_p")
        self.max_tokens = config_dict.get("max_output_tokens")
        self.batch_size = config_dict.get("batch_size", None)
        self.rpm = None            # subclass must set
        self.rate_limiter = None   # subclass must set

    def _create_client(self):
        """Create and return the async client. Subclass must implement."""
        raise NotImplementedError

    async def _do_request(self, client, messages):
        """Execute a single API request. Subclass must implement.

        Returns GenerationResult on success.
        Raises on transient errors (will be retried).
        """
        raise NotImplementedError

    def _get_retry_after(self, error):
        """Extract Retry-After seconds from an error/response, if available."""
        return None

    # ── retry wrapper ──────────────────────────────────────────────

    async def _retry_request(self, client, messages):
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            await self.rate_limiter.acquire()
            try:
                return await self._do_request(client, messages)
            except Exception as e:
                last_error = e
                # Check for rate limit (429-like)
                is_rate_limit = getattr(e, 'status_code', None) == 429
                if is_rate_limit:
                    self.rate_limiter.drain()
                    retry_after = self._get_retry_after(e)
                    wait = retry_after if retry_after else (2 ** attempt) * 5
                    print(f"  Rate limited (429), draining bucket, retrying in {wait}s...")
                else:
                    wait = (2 ** attempt) * 2
                    print(f"  Request error ({e}), retrying in {wait}s (attempt {attempt+1}/{self.MAX_RETRIES})...")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(wait)

        print(f"  Request failed after {self.MAX_RETRIES} attempts: {last_error}")
        return GenerationResult(text="", token_count=0)

    # ── public interface ───────────────────────────────────────────

    async def generate_one(self, client, messages):
        """Generate a single response within an existing async context."""
        return await self._retry_request(client, messages)

    def generate(self, batch_messages):
        return asyncio.run(self._generate_batch(batch_messages))

    async def _generate_batch(self, batch_messages):
        self.rate_limiter._lock = asyncio.Lock()
        client = self._create_client()
        try:
            tasks = [self._retry_request(client, msgs) for msgs in batch_messages]
            return await asyncio.gather(*tasks)
        finally:
            await self._cleanup_client(client)

    def run_workers(self, worker_fn, num_workers):
        return asyncio.run(self._run_workers(worker_fn, num_workers))

    async def _run_workers(self, worker_fn, num_workers):
        self.rate_limiter._lock = asyncio.Lock()
        client = self._create_client()
        try:
            tasks = [worker_fn(client) for _ in range(num_workers)]
            await asyncio.gather(*tasks)
        finally:
            await self._cleanup_client(client)

    async def _cleanup_client(self, client):
        """Close the client if it has an async close method."""
        if hasattr(client, 'aclose'):
            await client.aclose()
        elif hasattr(client, 'close'):
            close_result = client.close()
            if asyncio.iscoroutine(close_result):
                await close_result
