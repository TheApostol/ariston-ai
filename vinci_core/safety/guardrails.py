"""
Safety layer — input/output validation + audit logging.
- Definitive diagnosis blocking (from ariston-ai-1)
- Uncertainty quantification (from Claude2.pdf spec)
- High-risk domain flagging
- Confidence scoring
"""

import re
import logging
from datetime import datetime, timezone
from typing import Tuple, Dict, Any

logger = logging.getLogger("ariston.safety")

# Output patterns that make definitive clinical claims
DEFINITIVE_DIAGNOSIS_TERMS = [
    r"\byou have\b",
    r"\bmy diagnosis is\b",
    r"\bthis is definitely\b",
    r"\byou are suffering from\b",
    r"\bthe cause is\b",
]

FALLBACK_MESSAGE = (
    "This assessment has been flagged for safety review. "
    "Ariston AI cannot formally diagnose or prescribe treatment. "
    "Please consult a qualified healthcare professional for definitive guidance."
)

UNCERTAINTY_PHRASES = [
    "may ", "might ", "could ", "possibly", "potentially", "uncertain",
    "consult a", "differential diagnosis", "further testing", "i'm not sure",
]

HIGH_RISK_PATTERNS = [
    r"\bdosage\b", r"\bdose\b", r"\bprescri", r"\bdiagnos",
    r"\btreatment plan\b", r"\bsurgical\b", r"\bmedication\b",
    r"\bclinical trial\b", r"\bFDA approval\b",
]


class SafetyGuardrails:

    @classmethod
    def validate_input(cls, prompt: str) -> Tuple[bool, str, Dict[str, Any]]:
        if len(prompt.strip()) < 5:
            return False, "Input too short.", {"safety_flag": "INPUT_TOO_SHORT", "confidence": 0.0}
        confidence = min(0.95, 0.40 + (len(prompt) / 1000.0))
        return True, prompt, {"confidence": confidence, "safety_flag": "SAFE"}

    @classmethod
    def validate_output(cls, content: str) -> Tuple[bool, str, Dict[str, Any]]:
        lower = content.lower()
        for pattern in DEFINITIVE_DIAGNOSIS_TERMS:
            if re.search(pattern, lower):
                return False, FALLBACK_MESSAGE, {
                    "safety_flag": "DEFINITIVE_DIAGNOSIS_BLOCKED",
                    "confidence": 0.10,
                }
        return True, content, {"safety_flag": "SAFE"}


def check_safety(content: str) -> dict:
    """Full safety metadata dict attached to every AIResponse."""
    lower = content.lower()

    _, _, output_check = SafetyGuardrails.validate_output(content)
    uncertain = any(phrase in lower for phrase in UNCERTAINTY_PHRASES)
    high_risk = any(re.search(pat, lower) for pat in HIGH_RISK_PATTERNS)

    safety_result = {
        "flag": output_check.get("safety_flag", "SAFE"),
        "uncertain": uncertain,
        "high_risk_domain": high_risk,
        "requires_review": high_risk and not uncertain,
        "confidence": output_check.get("confidence", 0.90),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("[AUDIT] safety=%s | preview=%s", safety_result, content[:120].replace("\n", " "))
    return safety_result
