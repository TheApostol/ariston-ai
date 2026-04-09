import os
from google import genai
from typing import List

class VisionRadiologyAgent:
    """
    Vision Radiology Agent (Gen 2): Specialized for high-precision multimodal analysis.
    Grounds reports in standard clinical templates (Findings, Impression, Differential).
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None

    async def analyze_scan(self, prompt: str, images: List[str] = None) -> str:
        if not self.client:
             return "[Vision Simulation] Missing Gemini API Key. Scan interpretation bypassed."
        
        if not images:
             return "[Vision Warning] No diagnostic images provided for interpretation."

        try:
            from google.genai import types
            import base64
            
            contents = []
            system_instruction = (
                "ACT AS ARISTON OS SENIOR RADIOLOGIST (GEN 2).\n"
                "Provide an exhaustive multimodal interpretation using the Ariston Diagnostic Template:\n"
                "1. STUDY QUALITY & ACCESSION\n"
                "2. CLINICAL FINDINGS (Specific anatomical detail)\n"
                "3. RADIOLOGIC IMPRESSION (Primary diagnosis)\n"
                "4. CORRELATION (Correlate with PubMed/Patient Context)\n"
                "5. URGENCY SCORE (1-10)\n"
            )
            contents.append(system_instruction)
            contents.append(f"Patient Clinical Context: {prompt}")

            for img_data in images:
                if "," in img_data:
                    header, b64_str = img_data.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                else:
                    b64_str = img_data
                    mime_type = "image/png"

                image_bytes = base64.b64decode(b64_str)
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=contents
            )
            return f"[Ariston Vision Gen-2 Report]\n{response.text}"
        except Exception as e:
            return f"[Vision Error] Analysis failed: {str(e)}"

    def generate_saliency_map(self, image_b64: str) -> str:
        """
        Mock saliency map generation.
        Returns a 'detected pathology' coordinate set for the frontend.
        """
        # In a real system, this would use a grad-cam or attention-map model.
        # For Ariston OS demo, we return a simulated 'high-saliency' coordinate.
        return {
            "pathology_detected": True,
            "coordinates": {"x": 120, "y": 240, "radius": 45},
            "confidence": 0.89,
            "label": "SIMULATED_PATHOLOGY"
        }
