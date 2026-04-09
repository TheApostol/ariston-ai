"""
Composable Pipeline Base — Ariston AI.

Replaces if/else workflow dispatch with structured step-based pipelines.
Each step receives a context dict and returns an updated context dict.

Pattern:
    pipeline = Pipeline([step1, step2, step3])
    result = await pipeline.run(initial_context)

This enables:
- Clinical reasoning pipelines
- Regulatory analysis pipelines
- LATAM submission pipelines
- Future extension without modifying existing steps
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

logger = logging.getLogger("ariston.pipeline")


@dataclass
class PipelineContext:
    """Shared mutable context passed through each pipeline step."""
    prompt: str
    layer: str = "base"
    patient_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    final_content: str = ""


# Type alias for a pipeline step function
StepFn = Callable[[PipelineContext], Awaitable[PipelineContext]]


class PipelineStep:
    """Wraps a coroutine function as a named, trackable pipeline step."""

    def __init__(self, name: str, fn: StepFn):
        self.name = name
        self.fn = fn

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        try:
            ctx = await self.fn(ctx)
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            logger.info("[Pipeline] step=%s status=ok latency_ms=%s", self.name, elapsed)
        except Exception as e:
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            logger.error("[Pipeline] step=%s status=error error=%s latency_ms=%s", self.name, e, elapsed)
            ctx.errors.append({"step": self.name, "error": str(e)})
        return ctx


class Pipeline:
    """
    Executes a sequence of async pipeline steps.
    Each step can read and write to the shared PipelineContext.
    Steps are run sequentially — each receives the output of the previous.
    """

    def __init__(self, steps: list[PipelineStep], name: str = "pipeline"):
        self.steps = steps
        self.name = name

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("[Pipeline] name=%s steps=%d starting", self.name, len(self.steps))
        t0 = time.perf_counter()

        for step in self.steps:
            ctx = await step(ctx)
            # Short-circuit on hard errors (safety blocks)
            if ctx.metadata.get("abort_pipeline"):
                logger.warning(
                    "[Pipeline] name=%s aborted at step=%s reason=%s",
                    self.name, step.name, ctx.metadata.get("abort_reason", "unknown"),
                )
                break

        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        ctx.metadata["pipeline_name"] = self.name
        ctx.metadata["pipeline_steps"] = [s.name for s in self.steps]
        ctx.metadata["pipeline_latency_ms"] = elapsed
        ctx.metadata["pipeline_errors"] = ctx.errors

        logger.info(
            "[Pipeline] name=%s done latency_ms=%s errors=%d",
            self.name, elapsed, len(ctx.errors),
        )
        return ctx


def step(name: str):
    """Decorator to register a coroutine function as a PipelineStep."""
    def decorator(fn: StepFn) -> PipelineStep:
        return PipelineStep(name=name, fn=fn)
    return decorator
