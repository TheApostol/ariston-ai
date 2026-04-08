"""
HippoKron — clinical AI layer.

Handles clinical queries, patient context, and FHIR-ready data structures.

This layer sits on top of Vinci Core and is responsible for:
- Structuring clinical context
- Passing it to the orchestration engine
- Returning enriched responses

Future:
- Integrate FHIR Patient / Observation / Condition resources
- Add clinical workflows (diagnostic reasoning, safety checks)
"""

from typing import Dict, Any, Optional
from vinci_core.engine import engine


def build_clinical_context(patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build structured clinical context.
    This will later map to FHIR resources.
    """
    context: Dict[str, Any] = {}

    if patient_context:
        context["patient"] = patient_context

    return context


async def clinical_query(
    prompt: str,
    patient_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main entry point for HippoKron clinical queries.
    """

    context = build_clinical_context(patient_context)

    response = await engine.run(
        prompt=prompt,
        context=context,
        layer="clinical"
    )

    return response
