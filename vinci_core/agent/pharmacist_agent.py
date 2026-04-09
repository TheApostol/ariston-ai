"""
PharmacistAgent — Drug-drug interactions, GxP label compliance, and pharmacovigilance.

Uses Gemini for fast, cost-efficient pharmacological review.
"""

import logging
from typing import Dict

from vinci_core.models.gemini_model import GeminiModel

logger = logging.getLogger("ariston.agents.pharmacist")


class PharmacistAgent:
    """
    Specialized agent for drug-drug interactions, GxP label compliance,
    and pharmacovigilance (adverse reactions).
    """

    def __init__(self):
        self.model = GeminiModel()

    async def review_medications(self, prompt: str, context: Dict) -> str:
        pharmacist_prompt = (
            "You are the Ariston OS Clinical Pharmacist Agent. Your role is to review "
            "medication lists for drug-drug interactions (DDI), contraindications, "
            "and adverse reaction risks grounded in RxNorm and OpenFDA.\n\n"
            f"Context: {context.get('pharma_grounding', 'None provided')}\n"
            f"Query: {prompt}\n\n"
            "Provide a highly detailed PharmD-grade assessment."
        )

        try:
            result = await self.model.generate(prompt=pharmacist_prompt)
            return result.get("content", "No content returned from pharmacist review.")
        except Exception as exc:
            logger.error(
                '{"event":"pharmacist_error","error":"%s"}', type(exc).__name__
            )
            return "Pharmacist review temporarily unavailable. Please retry."

