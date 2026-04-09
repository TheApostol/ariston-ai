"""
Composable Step Pipelines for Vinci Core.

Each pipeline is a list of async step functions.
Every step receives a context dict and returns an updated context dict.
Steps record errors in ctx["_step_errors"] without aborting the pipeline.

Built-in pipelines:
  - clinical_pipeline_steps   — Extraction → Ontology Grounding → Consensus
  - regulatory_pipeline_steps — Jurisdiction → Risk → Compliance Draft

Usage:
    from vinci_core.routing.pipeline import run_pipeline, clinical_pipeline_steps

    ctx = {"prompt": "...", "layer": "clinical"}
    result = await run_pipeline(clinical_pipeline_steps, ctx)
"""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

logger = logging.getLogger("ariston.pipeline")

StepFn = Callable[[dict], Awaitable[dict]]


async def run_pipeline(steps: list[StepFn], ctx: dict) -> dict:
    """
    Execute a list of async step functions sequentially.

    Each step receives the accumulated context and returns it updated.
    If a step raises, the error is recorded in ctx["_step_errors"] and
    execution continues with the next step.
    """
    ctx.setdefault("_step_errors", [])
    for step in steps:
        step_name = getattr(step, "__name__", repr(step))
        try:
            ctx = await step(ctx)
            logger.debug("[pipeline] step=%s completed", step_name)
        except Exception as exc:
            logger.warning("[pipeline] step=%s raised %s", step_name, exc)
            ctx["_step_errors"].append({"step": step_name, "error": str(exc)})
    return ctx


# ---------------------------------------------------------------------------
# Clinical reasoning pipeline
# ---------------------------------------------------------------------------

async def _extract_entities(ctx: dict) -> dict:
    """Step 1 — Extract medical entities from the prompt."""
    from vinci_core.engine import engine

    res = await engine.run(
        prompt=(
            "Extract all medical symptoms, conditions, and drugs from this text. "
            f"Return as a comma-separated list only. Text: '{ctx['prompt']}'"
        ),
        layer="data",
        use_rag=False,
    )
    raw = [e.strip() for e in res.content.replace("\n", " ").split(",") if e.strip()]
    ctx["extracted_entities"] = raw
    return ctx


async def _ground_ontology(ctx: dict) -> dict:
    """Step 2 — Map extracted entities to standard clinical ontologies."""
    from vinci_core.tools.ontology import ontology_mapper

    entities = ctx.get("extracted_entities", [])
    grounded = ontology_mapper.ground_entities(entities)
    ctx["grounded_entities"] = grounded
    ctx["grounded_str"] = "\n".join(
        f"- {e['term']} ({e['system']}: {e['code']})" for e in grounded
    )
    return ctx


async def _consensus_synthesis(ctx: dict) -> dict:
    """Step 3 — Run clinical consensus synthesis with grounded entities."""
    from vinci_core.engine import engine

    grounded_str = ctx.get("grounded_str", "")
    prompt = ctx.get("prompt", "")
    structured = (
        "STRUCTURED CLINICAL REVIEW PIPELINE\n"
        "-----------------------------------\n"
        f"Grounded Entities:\n{grounded_str}\n"
        f"Original Query: {prompt}\n"
        "-----------------------------------\n"
        "Instructions:\n"
        "1. Evaluate the grounded ontological entities.\n"
        "2. Consider provided tool/literature evidence.\n"
        "3. Output a structured, safely-hedged clinical evaluation."
    )

    res = await engine.run(
        prompt=structured,
        layer="clinical",
        patient_id=ctx.get("patient_id"),
    )
    ctx["final_response"] = res
    return ctx


clinical_pipeline_steps: list[StepFn] = [
    _extract_entities,
    _ground_ontology,
    _consensus_synthesis,
]


# ---------------------------------------------------------------------------
# Regulatory analysis pipeline
# ---------------------------------------------------------------------------

async def _identify_jurisdiction(ctx: dict) -> dict:
    """Step 1 — Detect the regulatory jurisdiction from the prompt."""
    from vinci_core.engine import engine

    res = await engine.run(
        prompt=(
            "Identify the primary regulatory jurisdiction (e.g., FDA, EMA, PMDA, Health Canada) "
            f"referenced in this text. Return only the agency name. Text: '{ctx['prompt']}'"
        ),
        layer="data",
        use_rag=False,
    )
    ctx["jurisdiction"] = res.content.strip()
    return ctx


async def _assess_regulatory_risk(ctx: dict) -> dict:
    """Step 2 — Assess regulatory risk level for the query."""
    from vinci_core.engine import engine

    jurisdiction = ctx.get("jurisdiction", "unknown")
    res = await engine.run(
        prompt=(
            f"Given the regulatory context ({jurisdiction}), assess the risk level "
            f"(HIGH / MEDIUM / LOW) for the following: '{ctx['prompt']}'. "
            "Return only: RISK: <level> | RATIONALE: <one sentence>"
        ),
        layer="pharma",
        use_rag=False,
    )
    ctx["risk_assessment"] = res.content.strip()
    return ctx


async def _draft_compliance_response(ctx: dict) -> dict:
    """Step 3 — Draft a structured regulatory compliance response."""
    from vinci_core.engine import engine

    jurisdiction = ctx.get("jurisdiction", "unknown")
    risk = ctx.get("risk_assessment", "")
    res = await engine.run(
        prompt=(
            f"You are a regulatory affairs expert.\n"
            f"Jurisdiction: {jurisdiction}\n"
            f"Risk: {risk}\n"
            f"Query: {ctx['prompt']}\n\n"
            "Provide a structured compliance response including:\n"
            "1. REGULATORY FRAMEWORK\n"
            "2. APPLICABLE REQUIREMENTS\n"
            "3. RECOMMENDED ACTIONS\n"
            "4. RISK MITIGATION"
        ),
        layer="pharma",
    )
    ctx["final_response"] = res
    return ctx


regulatory_pipeline_steps: list[StepFn] = [
    _identify_jurisdiction,
    _assess_regulatory_risk,
    _draft_compliance_response,
]
