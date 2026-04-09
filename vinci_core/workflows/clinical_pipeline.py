from vinci_core.schemas import AIRequest, AIResponse


class ClinicalPipeline:
    """
    Implements unstructured-to-structured decision pipelines (inspired by Isaree.ai).
    Breaks a single prompt into: Extraction -> Routing -> Consensus Synthesis -> Formatting.
    """
    def __init__(self, engine):
        self.engine = engine

    async def execute(self, request: AIRequest) -> AIResponse:
        original_prompt = request.prompt
        
        # Explicit FHIR Interoperability
        fhir_context = ""
        if request.fhir_bundle:
            fhir_context = "FHIR Clinical Context:\\n"
            for resource in request.fhir_bundle:
                r_type = resource.get("resourceType")
                status = resource.get("status", "unknown")
                
                if r_type == "Observation":
                    code_display = resource.get("code", {}).get("text", "Unknown code")
                    val_q = resource.get("valueQuantity", {})
                    val_s = resource.get("valueString", "")
                    if val_s:
                        fhir_context += f"- [OBSERVATION: {status.upper()}] {code_display}: {val_s}\\n"
                    elif val_q:
                        fhir_context += f"- [OBSERVATION: {status.upper()}] {code_display}: {val_q.get('value')} {val_q.get('unit')}\\n"
                
                elif r_type == "Condition":
                    code_display = resource.get("code", {}).get("text", "Unknown condition")
                    clinical_status = resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "unknown")
                    fhir_context += f"- [CONDITION: {clinical_status.upper()}] {code_display}\\n"
                
                elif r_type == "DiagnosticReport":
                    code_display = resource.get("code", {}).get("text", "Unknown report")
                    fhir_context += f"- [DIAGNOSTIC REPORT: {status.upper()}] {code_display}\\n"
            
            original_prompt = f"{fhir_context}\\n\\n[NEW QUERY]\\n{original_prompt}"
        
        # Step 1: Data Extraction
        extraction_req = AIRequest(
            prompt=f"Extract all medical symptoms and drugs from this text. Return as comma-separated list. Text: '{original_prompt}'",
            context={"layer": "data"}
        )
        extraction_res = await self.engine.run(extraction_req)
        raw_entities = extraction_res.content.replace('\n', ' ').split(',')
        
        # Step 1.5: Semantic Grounding (Ontology mapping)
        from vinci_core.tools.ontology import ontology_mapper
        grounded_entities = ontology_mapper.ground_entities(raw_entities)
        grounded_str = "\n".join([f"- {e['term']} ({e['system']}: {e['code']})" for e in grounded_entities])

        # Step 2 & 3: Tool Retrieval + Consensus
        structured_prompt = (
            f"STRUCTURED CLINICAL REVIEW PIPELINE\n"
            f"-----------------------------------\n"
            f"Grounded Entities:\n{grounded_str}\n"
            f"Original Query: {original_prompt}\n"
            f"-----------------------------------\n"
            f"Instructions:\n"
            f"1. Evaluate the grounded ontological entities.\n"
            f"2. Consider provided tool/literature evidence.\n"
            f"3. Output a structured, safely-hedged evaluation."
        )
        
        final_req = AIRequest(
            prompt=structured_prompt,
            context={"layer": "clinical"} 
        )
        
        final_res = await self.engine.run(final_req)
        
        # Append pipeline metadata
        final_res.metadata["pipeline"] = "Ariston LifeScience OS Path"
        final_res.metadata["grounded_entities"] = grounded_entities
        
        return final_res
