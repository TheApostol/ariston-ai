from typing import Tuple, Dict, Any
import re

class SafetyGuardrails:
    # Danger words that sound definitive
    DEFINITIVE_DIAGNOSIS_TERMS = [
        r"\byou have\b",
        r"\bmy diagnosis is\b",
        r"\bthis is definitely\b",
        r"\byou are suffering from\b",
        r"\byou are experiencing\b", # borderline, but commonly restricted
        r"\bthe cause is\b"
    ]
    
    # Generic safe fallback wording
    FALLBACK_MESSAGE = (
        "I'm sorry, returning this assessment encountered a safety flag regarding medical language. "
        "Please remember I am an AI and cannot formally diagnose or prescribe treatment. "
        "Consult a qualified healthcare professional for definitive guidance."
    )

    @classmethod
    def validate_input(cls, prompt: str) -> Tuple[bool, str, Dict[str, Any]]:
        if len(prompt.strip()) < 5:
            return False, "Input too short for meaningful analysis.", {"safety_flag": "INPUT_TOO_SHORT", "confidence": 0.0}
        
        # Simple heuristic for confidence based on length/complexity
        confidence = min(0.95, 0.40 + (len(prompt) / 1000.0))
        return True, prompt, {"confidence": confidence, "safety_flag": "SAFE"}

    @classmethod
    def validate_output(cls, content: str) -> Tuple[bool, str, Dict[str, Any]]:
        # Check against definitive diagnosis
        content_lower = content.lower()
        for pattern in cls.DEFINITIVE_DIAGNOSIS_TERMS:
            if re.search(pattern, content_lower):
                return False, cls.FALLBACK_MESSAGE, {"safety_flag": "DEFINITIVE_DIAGNOSIS_BLOCKED", "confidence": 0.10}

        return True, content, {"safety_flag": "SAFE"}
