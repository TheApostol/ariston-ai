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
            fhir_context = "FHIR Observations Context:\\n"
            for resource in request.fhir_bundle:
                if resource.get("resourceType") == "Observation":
                    status = resource.get("status", "unknown")
                    code_display = resource.get("code", {}).get("text", "Unknown code")
                    val_q = resource.get("valueQuantity", {})
                    val_s = resource.get("valueString", "")
                    
                    if val_s:
                        fhir_context += f"- [{status.upper()}] {code_display}: {val_s}\\n"
                    elif val_q:
                        fhir_context += f"- [{status.upper()}] {code_display}: {val_q.get('value')} {val_q.get('unit')}\\n"
            
            original_prompt = f"{fhir_context}\\n\\n[NEW QUERY]\\n{original_prompt}"
        
        # Step 1: Data Extraction (uses cheaper "data" layer model)
        extraction_req = AIRequest(
            prompt=f"Extract all medical symptoms and drugs from this text. Return as comma-separated list. Text: '{original_prompt}'",
            context={"layer": "data"}
        )
        extraction_res = await self.engine.run(extraction_req)
        extracted_entities = extraction_res.content.replace('\n', ' ')

        # Step 2 & 3: Tool Retrieval + Consensus
        # We wrap the data into a formalized system prompt, then route to the "clinical" 
        # layer which automatically runs PubMed search AND the multi-model ConsensusRouter.
        structured_prompt = (
            f"STRUCTURED CLINICAL REVIEW PIPELINE\n"
            f"-----------------------------------\n"
            f"Extracted Entities: {extracted_entities}\n"
            f"Original Query: {original_prompt}\n"
            f"-----------------------------------\n"
            f"Instructions:\n"
            f"1. Evaluate the entities.\n"
            f"2. Consider provided tool/literature evidence.\n"
            f"3. Output a structured, safely-hedged evaluation."
        )
        
        final_req = AIRequest(
            prompt=structured_prompt,
            context={"layer": "clinical"} # Triggers ConsensusRouter and MedicalTools
        )
        
        final_res = await self.engine.run(final_req)
        
        # Append pipeline metadata
        final_res.metadata["pipeline"] = "Isaree.ai Structured Clinical Pipeline"
        final_res.metadata["extracted_entities"] = extracted_entities
        
        return final_res
