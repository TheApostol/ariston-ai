"""
Intent Classifier — autonomous layer routing before execution.
Classifies prompts into pharma/clinical/data/radiology/general.

Two-stage: fast keyword heuristics → LLM fallback (OpenRouter, free tier).
"""

import logging

from vinci_core.models.openrouter_model import OpenRouterModel

logger = logging.getLogger("ariston.classifier")

VALID_LAYERS = ["clinical", "pharma", "data", "radiology", "general"]

VISION_KEYWORDS   = ["x-ray", "mri", "ct scan", "image", "scan", "radiograph", "ultrasound", "pet scan"]
PHARMA_KEYWORDS   = ["drug", "medication", "dosage", "pharmacology", "interaction", "fda", "regulatory", "ectd", "csr", "nda", "bla"]
DATA_KEYWORDS     = ["dataset", "csv", "cohort", "rwe", "real-world", "biomarker", "signal", "pharmacovigilance"]
CLINICAL_KEYWORDS = ["symptom", "diagnosis", "patient", "disease", "treatment", "clinical", "therapy", "prognosis"]


class IntentClassifier:
    """
    Two-stage classifier: fast heuristic → LLM fallback.
    Uses OpenRouter (free) to keep cost near zero.
    """

    def __init__(self):
        self.brain = OpenRouterModel()

    async def classify(self, prompt: str) -> str:
        lower = prompt.lower()

        # Stage 1: fast keyword heuristics
        if any(w in lower for w in VISION_KEYWORDS):
            return "radiology"
        if any(w in lower for w in PHARMA_KEYWORDS):
            return "pharma"
        if any(w in lower for w in DATA_KEYWORDS):
            return "data"
        if any(w in lower for w in CLINICAL_KEYWORDS):
            return "clinical"

        # Stage 2: LLM classification
        system = (
            "You are the routing brain of a Life Sciences AI platform. "
            "Classify the user prompt into exactly one category:\n"
            "- 'clinical': symptoms, diagnosis, patient care, diseases, treatment\n"
            "- 'pharma': drugs, pharmacology, FDA submissions, regulatory documents\n"
            "- 'data': datasets, real-world evidence, biomarkers, signal detection\n"
            "- 'radiology': medical images, scans, X-rays, MRI\n"
            "- 'general': greetings, unrelated to medicine\n\n"
            "Reply with ONLY the category string. No explanation.\n\n"
            f"Prompt: {prompt}"
        )
        try:
            result = await self.brain.generate(messages=[{"role": "user", "content": system}])
            content = result.get("content", "") if isinstance(result, dict) else str(result)
            content = content.lower().strip()
            for layer in VALID_LAYERS:
                if layer in content:
                    logger.debug('{"event":"classifier_llm","result":"%s"}', layer)
                    return layer
        except Exception as exc:
            logger.warning(
                '{"event":"classifier_llm_failed","error":"%s","fallback":"general"}',
                type(exc).__name__,
            )

        return "general"


classifier = IntentClassifier()

