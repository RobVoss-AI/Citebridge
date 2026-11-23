"""
Client-side rate limiting and retry for the Gemini Notebook API.

Documented limits (per user): UploadSource 20/min, CreateSource 20/min,
every other method 60/min. Per project: 100 reads/min and 100 writes/min.
Exceeding them returns 429 RESOURCE_EXHAUSTED with a google.rpc.RetryInfo
``retryDelay`` (e.g. "3s"). We pace ourselves below the limits and, when a
429 still happens, honour retryDelay before falling back to exponential
backoff with jitter.
"""

from __future__ import annotations

import random
import re
import threading
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


class TokenBucket:
    """Simple thread-safe token bucket: ``rate`` tokens per ``per`` seconds."""

    def __init__(self, rate: int, per: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.capacity = float(rate)
        self.tokens = float(rate)
        self.fill_rate = rate / per
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.fill_rate)
        self._last = now

    def acquire(self) -> None:
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.fill_rate
            self._sleep(wait)


class RetryableError(Exception):
    """An error worth retrying. ``retry_after`` (seconds) comes from RetryInfo."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


def parse_duration(value: Optional[str]) -> Optional[float]:
    """Parse a protobuf Duration string like '3s' or '1.5s'."""
    if not value:
        return None
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)s\s*", str(value))
    return float(m.group(1)) if m else None


def with_retries(fn: Callable[[], T], *, attempts: int = 6, base: float = 1.0,
                 cap: float = 60.0, sleep: Callable[[float], None] = time.sleep,
                 on_retry: Optional[Callable[[int, float, Exception], None]] = None) -> T:
    """Call ``fn``; on RetryableError wait (RetryInfo first, else backoff) and retry."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts:
                raise
            if exc.retry_after is not None:
                delay = exc.retry_after + random.uniform(0, 0.5)
            else:
                delay = min(cap, base * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            if on_retry:
                on_retry(attempt, delay, exc)
            sleep(delay)
    raise AssertionError("unreachable")


class ApiLimits:
    """One set of buckets per Google account (limits are per user)."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        # Stay a little under the documented ceilings.
        self.upload = TokenBucket(18, clock=clock, sleep=sleep)
        self.create_source = TokenBucket(18, clock=clock, sleep=sleep)
        self.other = TokenBucket(55, clock=clock, sleep=sleep)
