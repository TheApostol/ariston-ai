import pytest
from vinci_core.safety.guardrails import SafetyGuardrails

def test_input_too_short():
    is_safe, msg, meta = SafetyGuardrails.validate_input("hi")
    assert not is_safe
    assert meta["safety_flag"] == "INPUT_TOO_SHORT"

def test_input_valid():
    is_safe, msg, meta = SafetyGuardrails.validate_input("I have a headache and some nausea.")
    assert is_safe
    assert meta["safety_flag"] == "SAFE"

def test_output_definitive_diagnosis_blocked():
    unsafe_content = "Based on these symptoms, my diagnosis is that you have a severe strain."
    is_safe, msg, meta = SafetyGuardrails.validate_output(unsafe_content)
    assert not is_safe
    assert "cannot formally diagnose" in msg
    assert meta["safety_flag"] == "DEFINITIVE_DIAGNOSIS_BLOCKED"

def test_output_safe():
    safe_content = "These symptoms could be consistent with a common cold, but you should see a doctor."
    is_safe, msg, meta = SafetyGuardrails.validate_output(safe_content)
    assert is_safe
    assert meta["safety_flag"] == "SAFE"
