"""
Continuous Loop Scheduler — autonomous improvement loop runner.

Keeps a background asyncio task that fires improvement cycles on a
configurable interval (default 60 minutes).  The loop:

  1. Analyzes benchmark logs for low-scoring patterns
  2. Reads unprocessed customer feedback signals
  3. Generates an improvement plan via LLM
  4. Logs the plan + marks signals processed

Thread-safe: uses a single asyncio.Lock to prevent concurrent cycles.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from vinci_core.continuous_improvement.improvement_agent import run_improvement_cycle

logger = logging.getLogger(__name__)

_SENTINEL = object()          # unique object used to detect "not set"

# ── Shared loop state ─────────────────────────────────────────────────────────

class _LoopState:
    def __init__(self):
        self.running: bool = False
        self.interval_seconds: int = 3600   # default: 1 hour
        self.cycles_completed: int = 0
        self.cycles_failed: int = 0
        self.last_run_at: Optional[str] = None
        self.last_status: str = "idle"
        self.next_run_at: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._lock: asyncio.Lock = asyncio.Lock()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "interval_seconds": self.interval_seconds,
            "cycles_completed": self.cycles_completed,
            "cycles_failed": self.cycles_failed,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "next_run_at": self.next_run_at,
        }


_state = _LoopState()


# ── Loop worker ───────────────────────────────────────────────────────────────

async def _loop_worker():
    """Background coroutine — runs improvement cycles on _state.interval_seconds cadence."""
    logger.info("[LoopScheduler] Continuous improvement loop started.")
    while _state.running:
        _state.last_status = "running"
        _state.last_run_at = datetime.now(timezone.utc).isoformat()
        async with _state._lock:
            try:
                result = await run_improvement_cycle()
                _state.cycles_completed += 1
                _state.last_status = result.get("status", "ok")
                logger.info(
                    "[LoopScheduler] Cycle #%d completed — status: %s",
                    _state.cycles_completed,
                    _state.last_status,
                )
            except Exception as e:
                _state.cycles_failed += 1
                _state.last_status = "error"
                logger.warning("[LoopScheduler] Cycle failed: %s", e)

        if not _state.running:
            break

        # Schedule next run
        next_run = (
            datetime.now(timezone.utc).timestamp() + _state.interval_seconds
        )
        _state.next_run_at = datetime.fromtimestamp(next_run, tz=timezone.utc).isoformat()

        # Sleep in small chunks so we can respond to stop() quickly
        elapsed = 0
        while elapsed < _state.interval_seconds and _state.running:
            await asyncio.sleep(min(5, _state.interval_seconds - elapsed))
            elapsed += 5

    _state.running = False
    _state.last_status = "stopped"
    _state.next_run_at = None
    logger.info("[LoopScheduler] Continuous improvement loop stopped.")


# ── Public API ────────────────────────────────────────────────────────────────

def start_loop(interval_seconds: int = 3600) -> Dict[str, Any]:
    """
    Start the continuous improvement loop.

    Args:
        interval_seconds: seconds between cycles (minimum 60)

    Returns:
        Current loop status snapshot.
    """
    if _state.running:
        return {"started": False, "reason": "already_running", **_state.snapshot()}

    interval_seconds = max(60, interval_seconds)
    _state.interval_seconds = interval_seconds
    _state.running = True

    # Create and schedule the background task
    loop = asyncio.get_event_loop()
    _state._task = loop.create_task(_loop_worker())

    logger.info(
        "[LoopScheduler] Loop started — interval: %ds", interval_seconds
    )
    return {"started": True, **_state.snapshot()}


def stop_loop() -> Dict[str, Any]:
    """
    Signal the loop to stop after the current cycle completes.

    Returns:
        Current loop status snapshot.
    """
    if not _state.running:
        return {"stopped": False, "reason": "not_running", **_state.snapshot()}

    _state.running = False
    logger.info("[LoopScheduler] Stop signal sent.")
    return {"stopped": True, **_state.snapshot()}


def get_loop_status() -> Dict[str, Any]:
    """Return the current loop status snapshot."""
    return _state.snapshot()


async def run_one_cycle() -> Dict[str, Any]:
    """
    Manually trigger a single improvement cycle outside the scheduler.
    Thread-safe — waits for any running cycle to finish first.

    Returns:
        Improvement cycle result dict.
    """
    async with _state._lock:
        _state.last_run_at = datetime.now(timezone.utc).isoformat()
        try:
            result = await run_improvement_cycle()
            _state.cycles_completed += 1
            _state.last_status = result.get("status", "ok")
            return result
        except Exception as e:
            _state.cycles_failed += 1
            _state.last_status = "error"
            return {"status": "error", "message": str(e)}
