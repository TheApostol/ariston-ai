"""
Composable step-based pipeline framework for Ariston AI.

Replaces ad-hoc if/else workflow dispatching with a clean pipeline pattern
where each step receives a context dict and returns an updated context dict.

Usage:
    pipeline = Pipeline([step_a, step_b, step_c])
    result   = await pipeline.run({"prompt": "...", "layer": "clinical"})

Each step is an async callable:
    async def my_step(ctx: PipelineContext) -> PipelineContext: ...

Pre-built pipelines:
    clinical_reasoning_pipeline   — entity extraction → grounding → consensus
    regulatory_analysis_pipeline  — doc classification → crosswalk → GxP report
"""

import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("ariston.pipeline")

# ── Types ─────────────────────────────────────────────────────────────────────

PipelineContext = Dict[str, Any]
StepFn = Callable[[PipelineContext], Coroutine[Any, Any, PipelineContext]]


# ── Core Pipeline ─────────────────────────────────────────────────────────────

class Pipeline:
    """
    Executes a sequence of async step functions, passing context through each.

    Errors in individual steps are caught and logged; the pipeline continues
    with the step marked as failed in `ctx["_step_errors"]`.
    """

    def __init__(self, steps: List[StepFn], name: str = "pipeline"):
        self.steps = steps
        self.name = name

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        ctx.setdefault("_step_errors", {})
        ctx.setdefault("_step_latencies_ms", {})

        logger.info('{"event":"pipeline_start","pipeline":"%s","steps":%d}', self.name, len(self.steps))
        total_start = time.monotonic()

        for step in self.steps:
            step_name = step.__name__
            step_start = time.monotonic()
            try:
                ctx = await step(ctx)
            except Exception as exc:
                logger.warning(
                    '{"event":"step_error","pipeline":"%s","step":"%s","error":"%s"}',
                    self.name, step_name, type(exc).__name__,
                )
                ctx["_step_errors"][step_name] = str(exc)
            finally:
                ctx["_step_latencies_ms"][step_name] = round(
                    (time.monotonic() - step_start) * 1000
                )

        total_ms = round((time.monotonic() - total_start) * 1000)
        logger.info(
            '{"event":"pipeline_complete","pipeline":"%s","total_ms":%d,"errors":%d}',
            self.name, total_ms, len(ctx["_step_errors"]),
        )
        return ctx


# ── Clinical Reasoning Pipeline ───────────────────────────────────────────────

async def _step_entity_extraction(ctx: PipelineContext) -> PipelineContext:
    """Extract medical entities from the prompt using the engine."""
    from vinci_core.engine import engine
    prompt = ctx.get("prompt", "")
    extraction_res = await engine.run(
        prompt=(
            "Extract all medical symptoms, conditions, and drugs from this text. "
            f"Return as a comma-separated list only. Text: '{prompt}'"
        ),
        layer="data",
        use_rag=False,
    )
    raw = extraction_res.content.replace("\n", " ")
    ctx["entities"] = [e.strip() for e in raw.split(",") if e.strip()]
    return ctx


async def _step_ontology_grounding(ctx: PipelineContext) -> PipelineContext:
    """Ground extracted entities against SNOMED/ICD-10/RxNorm ontologies."""
    entities = ctx.get("entities", [])
    try:
        from vinci_core.tools.ontology import ontology_mapper
        grounded = ontology_mapper.ground_entities(entities)
        ctx["grounded_entities"] = grounded
        ctx["grounded_str"] = "\n".join(
            f"- {e['term']} ({e['system']}: {e['code']})" for e in grounded
        )
    except Exception:
        ctx["grounded_entities"] = [{"term": e, "system": "raw", "code": "N/A"} for e in entities]
        ctx["grounded_str"] = "\n".join(f"- {e}" for e in entities)
    return ctx


async def _step_clinical_consensus(ctx: PipelineContext) -> PipelineContext:
    """Run consensus model on the grounded clinical query."""
    from vinci_core.engine import engine
    grounded_str = ctx.get("grounded_str", "")
    prompt = ctx.get("prompt", "")
    structured_prompt = (
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
        prompt=structured_prompt,
        layer="clinical",
        patient_id=ctx.get("patient_id"),
    )
    ctx["clinical_result"] = res.content
    ctx["clinical_model"] = res.model
    ctx["clinical_safety"] = res.metadata.get("safety", {}) if res.metadata else {}
    return ctx


#: End-to-end clinical reasoning pipeline
clinical_reasoning_pipeline = Pipeline(
    steps=[_step_entity_extraction, _step_ontology_grounding, _step_clinical_consensus],
    name="clinical_reasoning",
)


# ── Regulatory Analysis Pipeline ──────────────────────────────────────────────

async def _step_doc_classification(ctx: PipelineContext) -> PipelineContext:
    """Classify the regulatory document type and target agency."""
    from vinci_core.agent.classifier import classifier
    prompt = ctx.get("prompt", "")
    layer = await classifier.classify(prompt)
    ctx["detected_layer"] = layer

    # Simple keyword-based doc type detection
    lower = prompt.lower()
    if any(k in lower for k in ["nda", "new drug application"]):
        ctx["doc_type"] = "NDA"
    elif any(k in lower for k in ["ind", "investigational new drug"]):
        ctx["doc_type"] = "IND"
    elif any(k in lower for k in ["bla", "biologics"]):
        ctx["doc_type"] = "BLA"
    elif any(k in lower for k in ["ectd", "common technical document"]):
        ctx["doc_type"] = "eCTD"
    else:
        ctx["doc_type"] = "GENERAL"
    return ctx


async def _step_regulatory_crosswalk(ctx: PipelineContext) -> PipelineContext:
    """Map FDA requirement to LatAm regulatory equivalents."""
    doc_type = ctx.get("doc_type", "GENERAL")
    locale = ctx.get("locale", "pt-BR")
    try:
        from app.localization.regulatory_mapping import map_requirement, get_agency_for_locale
        agency = get_agency_for_locale(locale)
        mapping = map_requirement(doc_type, agency)
        ctx["regulatory_mapping"] = mapping
        ctx["target_agency"] = agency
    except Exception:
        ctx["regulatory_mapping"] = None
        ctx["target_agency"] = "UNKNOWN"
    return ctx


async def _step_gxp_report(ctx: PipelineContext) -> PipelineContext:
    """Generate a GxP-compliant regulatory report."""
    import uuid
    from vinci_core.agent.regulatory_agent import regulatory_copilot
    from vinci_core.engine import engine

    # Generate a pharma-layer analysis first
    prompt = ctx.get("prompt", "")
    analysis = await engine.run(prompt=prompt, layer="pharma", use_rag=True)
    ctx["regulatory_analysis"] = analysis.content

    # Sign with regulatory copilot
    job_id = ctx.get("job_id", str(uuid.uuid4()))
    gxp = regulatory_copilot.generate_report(
        job_id=job_id,
        prompt=prompt,
        result=analysis.content,
        audit_logs=[],
    )
    ctx["gxp_report"] = gxp
    return ctx


#: End-to-end regulatory analysis pipeline
regulatory_analysis_pipeline = Pipeline(
    steps=[_step_doc_classification, _step_regulatory_crosswalk, _step_gxp_report],
    name="regulatory_analysis",
)
