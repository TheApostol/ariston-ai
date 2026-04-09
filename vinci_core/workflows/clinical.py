"""
Clinical trial optimization workflow (Phase 1 product — Claude2.pdf).
- Protocol analysis and optimization
- Patient matching
- Trial feasibility assessment
"""

from typing import Optional
from vinci_core.engine import engine
from vinci_core.knowledge.retriever import retrieve, format_context
from vinci_core.tools.medical_tools import MedicalTools


async def optimize_trial_protocol(
    indication: str,
    phase: str,
    draft_protocol: Optional[str] = None,
    patient_population: Optional[dict] = None,
) -> dict:
    knowledge_chunks = await retrieve(
        f"{indication} phase {phase} clinical trial protocol", layer="clinical", max_results=6
    )
    comparable_trials = await MedicalTools.get_clinical_trials(f"{indication} phase {phase}")
    knowledge_text = format_context(knowledge_chunks)

    trial_summaries = "\n".join(
        f"- {t.get('title')} ({t.get('status')})" for t in comparable_trials
    )

    prompt = "\n\n".join(filter(None, [
        f"Analyze and optimize a Phase {phase} clinical trial protocol for: {indication}.",
        f"Draft Protocol:\n{draft_protocol}" if draft_protocol else None,
        f"Target Population:\n{patient_population}" if patient_population else None,
        f"Comparable Active Trials:\n{trial_summaries}" if trial_summaries else None,
        f"Clinical Evidence:\n{knowledge_text}" if knowledge_text else None,
        (
            "Provide: (1) Protocol strengths/weaknesses, "
            "(2) Inclusion/exclusion criteria optimization, "
            "(3) Primary/secondary endpoint recommendations, "
            "(4) Site selection criteria, "
            "(5) Recruitment feasibility assessment."
        ),
    ]))

    response = await engine.run(prompt=prompt, layer="clinical", use_rag=False)
    return {
        "indication": indication,
        "phase": phase,
        "recommendations": response.content,
        "comparable_trials": comparable_trials,
        "safety": response.metadata.get("safety", {}),
        "model": response.model,
    }


async def match_patients(trial_criteria: dict, patient_records: list) -> dict:
    prompt = (
        f"Evaluate patient records against trial eligibility criteria.\n\n"
        f"Trial Criteria:\n{trial_criteria}\n\n"
        f"Patients ({len(patient_records)}):\n"
        + "\n---\n".join(str(p) for p in patient_records[:20])
        + "\n\nFor each patient: ELIGIBLE / INELIGIBLE / BORDERLINE with rationale. "
        "Flag any screening risks."
    )
    response = await engine.run(prompt=prompt, layer="clinical", use_rag=False)
    return {
        "total_evaluated": len(patient_records),
        "matching_results": response.content,
        "safety": response.metadata.get("safety", {}),
    }
