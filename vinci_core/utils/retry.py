"""
Async retry decorator with exponential back-off.

Usage:
    @async_retry(max_attempts=3, base_delay=1.0)
    async def generate(...):
        ...

Transient errors (rate-limit 429, timeout, 502/503) trigger a retry.
Non-transient errors (auth failures, 400 bad-request) raise immediately.
"""

import asyncio
import logging
from functools import wraps

logger = logging.getLogger("ariston.retry")

_TRANSIENT_MARKERS = ("429", "timeout", "502", "503", "rate limit", "overloaded")


def async_retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator that retries an async callable on transient provider errors."""
    if max_attempts < 1:
        raise ValueError(f"async_retry: max_attempts must be >= 1, got {max_attempts}")

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    msg = str(exc).lower()
                    is_transient = any(m in msg for m in _TRANSIENT_MARKERS)
                    if not is_transient:
                        raise
                    last_exc = exc
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "[retry] %s attempt %d/%d failed (%s); retrying in %.1fs",
                            func.__qualname__, attempt, max_attempts, exc, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "[retry] %s exhausted %d attempts: %s",
                            func.__qualname__, max_attempts, exc,
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
