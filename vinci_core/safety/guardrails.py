"""
Safety layer — input/output validation + audit logging.

Covers:
- Definitive diagnosis blocking (clinical AI safety)
- Uncertainty quantification
- High-risk domain flagging
- LATAM regulatory language validation
- Confidence scoring
- Compliance-ready audit metadata
"""

import re
import logging
from datetime import datetime, timezone
from typing import Tuple, Dict, Any

logger = logging.getLogger("ariston.safety")

# Clinical: output patterns that make definitive claims
DEFINITIVE_DIAGNOSIS_TERMS = [
    r"\byou have\b",
    r"\bmy diagnosis is\b",
    r"\bthis is definitely\b",
    r"\byou are suffering from\b",
    r"\bthe cause is\b",
    r"\byou definitely have\b",
    r"\bthis confirms\b",
]

# LATAM: patterns that make unauthorized regulatory assertions
LATAM_DEFINITIVE_TERMS = [
    r"\bthis product is approved\b",
    r"\banvisa has approved\b",
    r"\bcofepris has approved\b",
    r"\binvima has approved\b",
    r"\banmat has approved\b",
    r"\bregulatory approval is guaranteed\b",
    r"\bwill be approved\b",
]

CLINICAL_FALLBACK_MESSAGE = (
    "This assessment has been flagged for safety review. "
    "Ariston AI cannot formally diagnose or prescribe treatment. "
    "Please consult a qualified healthcare professional for definitive guidance."
)

LATAM_FALLBACK_MESSAGE = (
    "This regulatory output has been flagged for review. "
    "Ariston AI provides regulatory intelligence support, not legal advice or regulatory decisions. "
    "Approval determinations are made exclusively by ANVISA, COFEPRIS, INVIMA, ANMAT, ISP, or other competent authorities. "
    "Consult a qualified regulatory affairs professional before taking action."
)

UNCERTAINTY_PHRASES = [
    "may ", "might ", "could ", "possibly", "potentially", "uncertain",
    "consult a", "differential diagnosis", "further testing", "i'm not sure",
    "estimated", "subject to", "pending", "subject to review", "recommend verifying",
]

HIGH_RISK_PATTERNS = [
    r"\bdosage\b", r"\bdose\b", r"\bprescri", r"\bdiagnos",
    r"\btreatment plan\b", r"\bsurgical\b", r"\bmedication\b",
    r"\bclinical trial\b", r"\bFDA approval\b",
]

LATAM_REGULATORY_PATTERNS = [
    r"\banvisa\b", r"\bcofepris\b", r"\binvima\b", r"\banmat\b",
    r"\bisp\b", r"\bregistro sanitario\b", r"\bpandrh\b",
    r"\bregistration dossier\b", r"\bregulatory submission\b",
]


class SafetyGuardrails:

    @classmethod
    def validate_input(cls, prompt: str) -> Tuple[bool, str, Dict[str, Any]]:
        stripped = prompt.strip()
        if len(stripped) < 5:
            return False, "Input too short.", {
                "safety_flag": "INPUT_TOO_SHORT",
                "confidence": 0.0,
            }
        # Confidence grows with prompt length (more context = more reliable)
        confidence = min(0.95, 0.40 + (len(stripped) / 1000.0))
        return True, prompt, {"confidence": confidence, "safety_flag": "SAFE"}

    @classmethod
    def validate_output(cls, content: str, layer: str = "base") -> Tuple[bool, str, Dict[str, Any]]:
        lower = content.lower()

        # Clinical definitive diagnosis check
        for pattern in DEFINITIVE_DIAGNOSIS_TERMS:
            if re.search(pattern, lower):
                logger.warning("[Safety] DEFINITIVE_DIAGNOSIS_BLOCKED layer=%s", layer)
                return False, CLINICAL_FALLBACK_MESSAGE, {
                    "safety_flag": "DEFINITIVE_DIAGNOSIS_BLOCKED",
                    "confidence": 0.10,
                }

        # LATAM regulatory unauthorized approval assertion check
        if layer == "latam":
            for pattern in LATAM_DEFINITIVE_TERMS:
                if re.search(pattern, lower):
                    logger.warning("[Safety] LATAM_APPROVAL_ASSERTION_BLOCKED layer=%s", layer)
                    return False, LATAM_FALLBACK_MESSAGE, {
                        "safety_flag": "LATAM_APPROVAL_ASSERTION_BLOCKED",
                        "confidence": 0.10,
                    }

        return True, content, {"safety_flag": "SAFE"}


def check_safety(content: str, layer: str = "base") -> dict:
    """Full safety metadata dict attached to every AIResponse."""
    lower = content.lower()

    _, _, output_check = SafetyGuardrails.validate_output(content, layer=layer)
    uncertain = any(phrase in lower for phrase in UNCERTAINTY_PHRASES)
    high_risk = any(re.search(pat, lower) for pat in HIGH_RISK_PATTERNS)
    latam_regulatory = any(re.search(pat, lower) for pat in LATAM_REGULATORY_PATTERNS)

    safety_result = {
        "flag": output_check.get("safety_flag", "SAFE"),
        "uncertain": uncertain,
        "high_risk_domain": high_risk,
        "latam_regulatory_content": latam_regulatory,
        "requires_review": (high_risk or latam_regulatory) and not uncertain,
        "confidence": output_check.get("confidence", 0.90),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    if safety_result["requires_review"]:
        logger.warning(
            "[AUDIT] requires_review=True layer=%s flag=%s | preview=%s",
            layer, safety_result["flag"], content[:120].replace("\n", " "),
        )
    else:
        logger.info(
            "[AUDIT] safety=%s | preview=%s",
            safety_result["flag"], content[:120].replace("\n", " "),
        )

    return safety_result
