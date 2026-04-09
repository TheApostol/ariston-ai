"""
Async retry utility with exponential back-off.

Usage:
    from vinci_core.utils.retry import async_retry

    @async_retry(max_attempts=3, base_delay=1.0, exceptions=(httpx.TimeoutException,))
    async def my_provider_call():
        ...
"""

import asyncio
import functools
import logging
from typing import Tuple, Type

logger = logging.getLogger("ariston.retry")


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
):
    """
    Decorator: retries an async function up to *max_attempts* times.

    Waits base_delay * backoff_factor^(attempt-1) seconds between retries.
    Raises the last exception if all attempts fail.

    Args:
        max_attempts:   Total number of attempts (1 = no retry).
        base_delay:     Initial delay in seconds before the first retry.
        backoff_factor: Multiplier applied to delay on each retry.
        exceptions:     Tuple of exception types to catch and retry on.
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc: BaseException = RuntimeError("No attempts made")
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    logger.warning(
                        "[retry] %s attempt %d/%d failed: %s — retrying in %.1fs",
                        fn.__qualname__, attempt, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
            logger.error(
                "[retry] %s failed after %d attempts: %s",
                fn.__qualname__, max_attempts, last_exc,
            )
            raise last_exc
        return wrapper
    return decorator
