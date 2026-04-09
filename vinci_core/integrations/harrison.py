"""
Harrison.ai integration — clinical-grade radiology and pathology AI.
Harrison.ai holds FDA clearances for chest X-ray, ECG, and pathology analysis.

STATUS: Partnership integration — requires commercial API agreement.
Contact: partnerships@harrison.ai

This module provides the interface layer so the integration is ready
the moment the partnership is signed. Swap HARRISON_API_KEY into .env
and set HARRISON_ENABLED=true to activate.

Capabilities (once live):
  - Chest X-ray analysis (pneumonia, nodules, effusions)
  - ECG interpretation
  - Pathology slide analysis
  - Structured radiology report generation
"""

import httpx
import os
from typing import Optional

HARRISON_API_URL = "https://api.harrison.ai/v1"
HARRISON_ENABLED = os.getenv("HARRISON_ENABLED", "false").lower() == "true"
HARRISON_API_KEY = os.getenv("HARRISON_API_KEY", "")


async def analyze_chest_xray(image_url: str, patient_context: Optional[dict] = None) -> dict:
    """
    Submit a chest X-ray for Harrison.ai analysis.
    Returns structured findings with confidence scores.
    """
    if not HARRISON_ENABLED:
        return _not_configured("chest_xray")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(
                f"{HARRISON_API_URL}/radiology/chest-xray",
                headers={"Authorization": f"Bearer {HARRISON_API_KEY}"},
                json={"image_url": image_url, "patient_context": patient_context or {}},
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e), "source": "harrison.ai"}


async def analyze_ecg(ecg_data: dict) -> dict:
    """Submit ECG data for Harrison.ai interpretation."""
    if not HARRISON_ENABLED:
        return _not_configured("ecg")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(
                f"{HARRISON_API_URL}/cardiology/ecg",
                headers={"Authorization": f"Bearer {HARRISON_API_KEY}"},
                json=ecg_data,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e), "source": "harrison.ai"}


def _not_configured(modality: str) -> dict:
    return {
        "status": "partnership_required",
        "modality": modality,
        "message": (
            "Harrison.ai integration requires a commercial API agreement. "
            "Set HARRISON_ENABLED=true and HARRISON_API_KEY in .env once signed."
        ),
        "contact": "partnerships@harrison.ai",
        "source": "harrison.ai",
    }
