"""
Clinical Pipeline — structured decision workflow.
Inspired by Isaree.ai: breaks a single prompt into
Extraction → Ontology Grounding → Consensus Synthesis.
"""

from typing import Optional, List, Dict, Any
from vinci_core.engine import engine


class ClinicalPipeline:
    """
    Implements unstructured-to-structured decision pipelines.
    Handles FHIR resource parsing, entity extraction, ontology grounding,
    and consensus synthesis.
    """

    @staticmethod
    def _parse_fhir_bundle(fhir_bundle: List[Dict[str, Any]]) -> str:
        """Convert a list of FHIR resources into a human-readable context string."""
        lines = ["FHIR Clinical Context:"]
        for resource in fhir_bundle:
            r_type = resource.get("resourceType")
            status = resource.get("status", "unknown")

            if r_type == "Observation":
                code_display = resource.get("code", {}).get("text", "Unknown code")
                val_q = resource.get("valueQuantity", {})
                val_s = resource.get("valueString", "")
                if val_s:
                    lines.append(f"- [OBSERVATION: {status.upper()}] {code_display}: {val_s}")
                elif val_q:
                    lines.append(
                        f"- [OBSERVATION: {status.upper()}] {code_display}: "
                        f"{val_q.get('value')} {val_q.get('unit')}"
                    )

            elif r_type == "Condition":
                code_display = resource.get("code", {}).get("text", "Unknown condition")
                clinical_status = (
                    resource.get("clinicalStatus", {})
                    .get("coding", [{}])[0]
                    .get("code", "unknown")
                )
                lines.append(f"- [CONDITION: {clinical_status.upper()}] {code_display}")

            elif r_type == "DiagnosticReport":
                code_display = resource.get("code", {}).get("text", "Unknown report")
                lines.append(f"- [DIAGNOSTIC REPORT: {status.upper()}] {code_display}")

        return "\n".join(lines)

    async def execute(
        self,
        prompt: str,
        fhir_bundle: Optional[List[Dict[str, Any]]] = None,
        patient_id: Optional[str] = None,
    ):
        """
        Run the full clinical pipeline.

        Args:
            prompt: the clinical query
            fhir_bundle: optional list of FHIR resources for context injection
            patient_id: optional patient ID for history injection in engine

        Returns:
            AIResponse from final consensus pass
        """
        enriched_prompt = prompt
        if fhir_bundle:
            fhir_context = self._parse_fhir_bundle(fhir_bundle)
            enriched_prompt = f"{fhir_context}\n\n[NEW QUERY]\n{prompt}"

        # Step 1: Entity extraction
        extraction_res = await engine.run(
            prompt=(
                f"Extract all medical symptoms, conditions, and drugs from this text. "
                f"Return as a comma-separated list only. Text: '{enriched_prompt}'"
            ),
            layer="data",
            use_rag=False,
        )
        raw_entities = [e.strip() for e in extraction_res.content.replace("\n", " ").split(",") if e.strip()]

        # Step 2: Ontology grounding
        try:
            from vinci_core.tools.ontology import ontology_mapper
            grounded_entities = ontology_mapper.ground_entities(raw_entities)
            grounded_str = "\n".join(
                [f"- {e['term']} ({e['system']}: {e['code']})" for e in grounded_entities]
            )
        except Exception:
            grounded_entities = [{"term": e, "system": "raw", "code": "N/A"} for e in raw_entities]
            grounded_str = "\n".join([f"- {e}" for e in raw_entities])

        # Step 3: Consensus synthesis with grounded context
        structured_prompt = (
            "STRUCTURED CLINICAL REVIEW PIPELINE\n"
            "-----------------------------------\n"
            f"Grounded Entities:\n{grounded_str}\n"
            f"Original Query: {enriched_prompt}\n"
            "-----------------------------------\n"
            "Instructions:\n"
            "1. Evaluate the grounded ontological entities.\n"
            "2. Consider provided tool/literature evidence.\n"
            "3. Output a structured, safely-hedged clinical evaluation."
        )

        final_res = await engine.run(
            prompt=structured_prompt,
            layer="clinical",
            patient_id=patient_id,
        )

        final_res.metadata["pipeline"] = "ClinicalPipeline"
        final_res.metadata["grounded_entities"] = grounded_entities

        return final_res


clinical_pipeline = ClinicalPipeline()
