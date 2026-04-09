import os
from typing import Dict, List
from vinci_core.models.gemini_model import GeminiModel

class PharmacistAgent:
    """
    Specialized agent for drug-drug interactions, GxP label compliance, 
    and pharmacovigilance (Adverse Reactions).
    """
    def __init__(self):
        self.model = GeminiModel()

    async def review_medications(self, prompt: str, context: Dict) -> str:
        # Enhance prompt with pharmacological constraints
        pharmacist_prompt = (
            "You are the Ariston OS Clinical Pharmacist Agent. Your role is to review "
            "medication lists for drug-drug interactions (DDI), contraindications, "
            "and adverse reaction risks grounded in RxNorm and OpenFDA.\n\n"
            f"Context: {context.get('pharma_grounding', 'None provided')}\n"
            f"Query: {prompt}\n\n"
            "Provide a highly detailed PharmD-grade assessment."
        )
        
        res = await self.model.generate({"prompt": pharmacist_prompt})
        return res.get("content", "Failed to generate pharmacy review.")
