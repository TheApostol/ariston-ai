import os
import base64
from typing import List


class VisionRadiologyAgent:
    """
    Vision Radiology Agent (Gen 2): Specialized for high-precision multimodal analysis.
    Grounds reports in standard clinical templates (Findings, Impression, Differential).
    google-genai imported lazily so startup doesn't fail when SDK isn't installed.
    """

    def __init__(self):
        self._client = None  # lazy-initialized on first use

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            from google import genai  # lazy import
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            pass
        return self._client

    async def analyze_scan(self, prompt: str, images: List[str] = None) -> str:
        client = self._get_client()

        if not client:
            # Graceful degradation — simulate a structured report
            return self._simulate_report(prompt, images)

        if not images:
            return "[Vision Warning] No diagnostic images provided for interpretation."

        try:
            from google.genai import types

            system_instruction = (
                "ACT AS ARISTON OS SENIOR RADIOLOGIST (GEN 2).\n"
                "Provide an exhaustive multimodal interpretation using the Ariston Diagnostic Template:\n"
                "1. STUDY QUALITY & ACCESSION\n"
                "2. CLINICAL FINDINGS (Specific anatomical detail)\n"
                "3. RADIOLOGIC IMPRESSION (Primary diagnosis)\n"
                "4. CORRELATION (Correlate with PubMed/Patient Context)\n"
                "5. URGENCY SCORE (1-10)\n"
            )

            contents = [system_instruction, f"Patient Clinical Context: {prompt}"]

            for img_data in images:
                if "," in img_data:
                    header, b64_str = img_data.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                else:
                    b64_str = img_data
                    mime_type = "image/png"

                image_bytes = base64.b64decode(b64_str)
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

            import asyncio as _asyncio
            response = await _asyncio.wait_for(
                _asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.0-flash",
                    contents=contents,
                ),
                timeout=60,
            )
            return f"[Ariston Vision Gen-2 Report]\n{response.text}"

        except Exception as e:
            return f"[Vision Error] Analysis failed: {str(e)}"

    def _simulate_report(self, prompt: str, images: List[str] = None) -> str:
        """Structured simulated report when Gemini API key is not configured."""
        n = len(images) if images else 0
        return (
            "[Ariston Vision — Simulation Mode (No GEMINI_API_KEY)]\n\n"
            f"1. STUDY QUALITY & ACCESSION\n"
            f"   Images received: {n}. Quality assessment pending live API.\n\n"
            f"2. CLINICAL FINDINGS\n"
            f"   Query: {prompt[:200]}\n"
            f"   Findings: Set GEMINI_API_KEY to enable live multimodal analysis.\n\n"
            f"3. RADIOLOGIC IMPRESSION\n"
            f"   Cannot generate impression without vision model access.\n\n"
            f"4. CORRELATION\n"
            f"   RAG-grounded correlation available via /analyze endpoint.\n\n"
            f"5. URGENCY SCORE\n"
            f"   N/A (simulation mode)"
        )

    def generate_saliency_map(self, image_b64: str) -> dict:
        return {
            "pathology_detected": True,
            "coordinates": {"x": 120, "y": 240, "radius": 45},
            "confidence": 0.89,
            "label": "SIMULATED_PATHOLOGY",
        }
