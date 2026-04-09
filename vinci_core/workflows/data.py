"""
Real-world evidence and pharmacovigilance signal detection workflow (Phase 2).
"""

from typing import Optional, List
from vinci_core.engine import engine
from vinci_core.knowledge.retriever import retrieve, format_context
from vinci_core.tools.medical_tools import MedicalTools


async def detect_safety_signals(
    drug_name: str,
    dataset_summary: Optional[dict] = None,
    time_period: Optional[str] = None,
) -> dict:
    faers_events = await MedicalTools.get_adverse_events(drug_name)
    knowledge_chunks = await retrieve(
        f"{drug_name} adverse events pharmacovigilance safety", layer="data", max_results=4
    )
    knowledge_text = format_context(knowledge_chunks)

    prompt = "\n\n".join(filter(None, [
        f"Perform pharmacovigilance signal detection for: {drug_name}",
        f"FAERS Top Adverse Events: {', '.join(faers_events)}" if faers_events else None,
        f"Internal Dataset:\n{dataset_summary}" if dataset_summary else None,
        f"Time Period: {time_period}" if time_period else None,
        f"Literature Evidence:\n{knowledge_text}" if knowledge_text else None,
        (
            "Provide: (1) Detected signals with disproportionality rationale, "
            "(2) Severity classification (critical/moderate/low), "
            "(3) Recommended regulatory actions (CIOMS/MedWatch), "
            "(4) Confidence and data limitations."
        ),
    ]))

    response = await engine.run(prompt=prompt, layer="data", use_rag=False)
    return {
        "drug_name": drug_name,
        "faers_events": faers_events,
        "signal_analysis": response.content,
        "safety": response.metadata.get("safety", {}),
        "model": response.model,
    }


async def generate_rwe_insights(
    research_question: str,
    dataset_description: Optional[str] = None,
    data_sources: Optional[List[str]] = None,
) -> dict:
    knowledge_chunks = await retrieve(
        research_question + " real world evidence epidemiology", layer="data", max_results=5
    )
    knowledge_text = format_context(knowledge_chunks)

    sources_text = ", ".join(data_sources) if data_sources else "not specified"
    prompt = "\n\n".join(filter(None, [
        f"Generate real-world evidence insights for:\n{research_question}",
        f"Dataset: {dataset_description}" if dataset_description else None,
        f"Data Sources: {sources_text}",
        f"Supporting Literature:\n{knowledge_text}" if knowledge_text else None,
        (
            "Structure as: (1) Key findings, (2) Effect sizes with CIs, "
            "(3) Confounders and limitations, (4) Regulatory/clinical implications, "
            "(5) Recommended follow-up analyses."
        ),
    ]))

    response = await engine.run(prompt=prompt, layer="data", use_rag=False)
    return {
        "research_question": research_question,
        "insights": response.content,
        "sources_used": len(knowledge_chunks),
        "safety": response.metadata.get("safety", {}),
        "model": response.model,
    }
