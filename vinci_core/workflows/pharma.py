"""
Regulatory Document Drafting Workflow — WEDGE PRODUCT.

This is the core revenue-generating product: AI-assisted drafting of
CSRs, eCTD modules, and FDA submission packages.

Demo flow:
  draft_regulatory_document(
      document_type="csr",
      drug_name="dabrafenib",
      indication="BRAF V600E metastatic melanoma",
      nct_id="NCT01227889"
  )

Grounded in live data from:
  - ClinicalTrials.gov (trial protocol, endpoints, eligibility)
  - PubMed (published results, safety literature)
  - ChEMBL (mechanism of action, pharmacology)
  - ICH E3 structure (CSR sections)
"""

import asyncio
from typing import Optional
from vinci_core.engine import engine
from vinci_core.knowledge.retriever import retrieve, format_context


# ICH E3-compliant CSR section structure
CSR_SECTIONS = [
    "1. Title Page",
    "2. Synopsis",
    "3. Table of Contents",
    "4. List of Abbreviations",
    "5. Ethics (IRB/IEC approval, GCP compliance)",
    "6. Investigators and Study Administrative Structure",
    "7. Introduction (disease background, rationale)",
    "8. Study Objectives (primary, secondary, exploratory)",
    "9. Investigational Plan (design, endpoints, sample size)",
    "10. Study Patients (eligibility, demographics, disposition)",
    "11. Efficacy Evaluation (primary/secondary endpoints, statistical analysis)",
    "12. Safety Evaluation (adverse events, laboratory data, vital signs)",
    "13. Discussion and Conclusions",
    "14. References",
    "Appendices",
]

DOCUMENT_TEMPLATES = {
    # ── Global / FDA ────────────────────────────────────────────────────────
    "csr": {
        "title": "Clinical Study Report (ICH E3)",
        "sections": CSR_SECTIONS,
        "guideline": "ICH E3 — Structure and Content of Clinical Study Reports",
        "fda_ref": "FDA Guidance: E3 Clinical Study Reports (2022)",
        "agency": "FDA / ICH",
        "language": "en",
    },
    "ectd": {
        "title": "eCTD Module 5 — Clinical Study Reports",
        "sections": ["Module 5.1 Table of Contents", "5.2 Tabular Listing of Clinical Studies",
                     "5.3.1 Reports of Biopharmaceutic Studies", "5.3.5 Reports of Efficacy and Safety Studies"],
        "guideline": "ICH M4E — Common Technical Document (CTD) Efficacy",
        "fda_ref": "FDA eCTD Specifications v3.2.2",
        "agency": "FDA / ICH",
        "language": "en",
    },
    "cmc": {
        "title": "Chemistry, Manufacturing and Controls (CMC) — Module 3",
        "sections": ["3.2.S Drug Substance", "3.2.P Drug Product", "3.2.A Appendices", "3.2.R Regional Information"],
        "guideline": "ICH Q8/Q9/Q10 — Pharmaceutical Development, Quality Risk Management",
        "fda_ref": "FDA Guidance: CMC Review during IND/NDA",
        "agency": "FDA / ICH",
        "language": "en",
    },
    "pv_narrative": {
        "title": "Pharmacovigilance Case Narrative (CIOMS)",
        "sections": ["Patient Information", "Event Description", "Drug Information",
                     "Clinical Course", "Assessment", "Reporter Information"],
        "guideline": "CIOMS VI — Management of Safety Information from Clinical Trials",
        "fda_ref": "FDA 21 CFR 312.32 — IND Safety Reports",
        "agency": "FDA / ICH",
        "language": "en",
    },

    # ── LATAM ────────────────────────────────────────────────────────────────
    "cofepris_registro": {
        "title": "Expediente de Registro Sanitario — COFEPRIS (México)",
        "sections": [
            "I. Solicitud y Datos del Titular",
            "II. Información del Medicamento (nombre, forma farmacéutica, vía de administración)",
            "III. Módulo Químico-Farmacéutico y Biológico (CTD Módulo 3)",
            "IV. Módulo No Clínico (farmacología, toxicología)",
            "V. Módulo Clínico — Resumen del Informe Clínico (ICH E3 adaptado)",
            "VI. Estudios de Bioequivalencia / Biodisponibilidad",
            "VII. Información de Seguridad y Farmacovigilancia (NOM-220-SSA1)",
            "VIII. Etiquetado y Prospecto (NOM-072-SSA1)",
            "IX. Declaración de Buenas Prácticas de Fabricación (BPF)",
            "X. Anexos y Documentación Complementaria",
        ],
        "guideline": "COFEPRIS — Reglamento de Insumos para la Salud / NOM-220-SSA1-2016",
        "fda_ref": "COFEPRIS Lineamientos para Registro de Medicamentos Alopáticos (2023)",
        "agency": "COFEPRIS",
        "language": "es",
    },
    "cofepris_pv": {
        "title": "Reporte de Caso de Farmacovigilancia — COFEPRIS (México)",
        "sections": [
            "1. Datos del Notificador",
            "2. Datos del Paciente",
            "3. Descripción de la Reacción Adversa",
            "4. Información del Medicamento Sospechoso",
            "5. Medicamentos Concomitantes",
            "6. Desenlace Clínico",
            "7. Evaluación de Causalidad (Algoritmo de Naranjo)",
        ],
        "guideline": "NOM-220-SSA1-2016 — Instalación y Operación de la Farmacovigilancia",
        "fda_ref": "COFEPRIS VIGIFARMA — Sistema Nacional de Farmacovigilancia",
        "agency": "COFEPRIS",
        "language": "es",
    },
    "anvisa_registro": {
        "title": "Dossiê de Registro de Medicamento Novo — ANVISA (Brasil)",
        "sections": [
            "1. Requerimentos Administrativos e Legais",
            "2. Módulo de Qualidade (CTD Módulo 3 / RDC 204/2017)",
            "3. Módulo Não Clínico — Farmacologia e Toxicologia",
            "4. Módulo Clínico — Relatório de Estudo Clínico (ICH E3)",
            "5. Bula do Produto (RDC 47/2009)",
            "6. Rotulagem (RDC 71/2009)",
            "7. Farmacovigilância — Plano de Gestão de Risco (PGR)",
            "8. Estudos de Biodisponibilidade/Bioequivalência (RDC 37/2011)",
            "9. Declaração de Boas Práticas de Fabricação (BPF/RDC 301/2019)",
        ],
        "guideline": "ANVISA RDC 204/2017 — Registro de Medicamentos Novos",
        "fda_ref": "ANVISA Resolução RDC 204/2017 e RDC 47/2009",
        "agency": "ANVISA",
        "language": "pt",
    },
    "anmat_registro": {
        "title": "Expediente de Autorización de Comercialización — ANMAT (Argentina)",
        "sections": [
            "I. Datos del Solicitante y del Producto",
            "II. Módulo de Calidad Farmacéutica (CTD Módulo 3 / Disposición ANMAT 3185/99)",
            "III. Módulo No Clínico (farmacología, toxicología preclínica)",
            "IV. Módulo Clínico — Informe de Ensayo Clínico (ICH E3 / Disposición 3311/10)",
            "V. Prospecto e Información para el Prescriptor",
            "VI. Rotulado (Disposición ANMAT 2819/04)",
            "VII. Plan de Gestión de Riesgos y Farmacovigilancia (Disposición 5358/12)",
            "VIII. Certificado de Buenas Prácticas de Fabricación (BPF)",
        ],
        "guideline": "ANMAT Disposición 3311/10 — Registro de Especialidades Medicinales",
        "fda_ref": "ANMAT Disposición 3311/10 y Disposición 5358/12",
        "agency": "ANMAT",
        "language": "es",
    },
    "invima_registro": {
        "title": "Expediente de Registro Sanitario — INVIMA (Colombia)",
        "sections": [
            "1. Información Administrativa del Titular y Fabricante",
            "2. Módulo de Calidad Farmacéutica (CTD Módulo 3 / Decreto 677/95)",
            "3. Estudios No Clínicos — Farmacología y Toxicología",
            "4. Estudios Clínicos — Informe de Estudio Clínico (Resolución 2378/08)",
            "5. Información del Producto para el Profesional de la Salud",
            "6. Información del Producto para el Paciente",
            "7. Farmacovigilancia — Plan de Minimización de Riesgos (Resolución 1403/07)",
            "8. Certificado de Cumplimiento de BPM",
        ],
        "guideline": "INVIMA Decreto 677/1995 y Resolución 2378/2008 — BPC",
        "fda_ref": "INVIMA Resolución 2378/2008 (Buenas Prácticas Clínicas)",
        "agency": "INVIMA",
        "language": "es",
    },
}


async def draft_regulatory_document(
    document_type: str,
    drug_name: str,
    indication: str,
    nct_id: Optional[str] = None,
    study_data: Optional[dict] = None,
    section: Optional[str] = None,
    language: Optional[str] = None,   # override template language: "en" | "es" | "pt"
) -> dict:
    """
    Main demo entry point. Produces a complete regulatory document draft.

    Args:
        document_type: csr | ectd | cmc | pv_narrative | cofepris_registro |
                       cofepris_pv | anvisa_registro | anmat_registro | invima_registro
        drug_name: e.g. "dabrafenib"
        indication: e.g. "BRAF V600E metastatic melanoma"
        nct_id: ClinicalTrials.gov ID for grounding (e.g. "NCT01227889")
        study_data: dict with any study-specific data to inject
        section: optional — draft only this section
        language: override output language ("en", "es", "pt"); defaults to template default
    """
    template = DOCUMENT_TEMPLATES.get(document_type.lower(), DOCUMENT_TEMPLATES["csr"])
    output_language = language or template.get("language", "en")

    # Parallel data fetch
    rag_query = f"{drug_name} {indication} {template['title']} regulatory"
    knowledge_chunks, trial_data = await asyncio.gather(
        retrieve(rag_query, layer="pharma", max_results=6),
        _fetch_trial_data(nct_id) if nct_id else asyncio.sleep(0, result={}),
    )
    knowledge_text = format_context(knowledge_chunks)

    # Build structured prompt
    prompt = _build_drafting_prompt(
        template=template,
        drug_name=drug_name,
        indication=indication,
        knowledge_text=knowledge_text,
        trial_data=trial_data,
        study_data=study_data or {},
        section=section,
        language=output_language,
    )

    response = await engine.run(
        prompt=prompt,
        layer="pharma",
        use_rag=False,  # already retrieved above
    )

    return {
        "document_type": template["title"],
        "drug_name": drug_name,
        "indication": indication,
        "nct_id": nct_id,
        "guideline": template["guideline"],
        "fda_reference": template["fda_ref"],
        "section_drafted": section or "full document",
        "draft": response.content,
        "sources_used": len(knowledge_chunks),
        "trial_data_used": bool(trial_data),
        "safety": response.metadata.get("safety", {}),
        "model": response.model,
        "job_id": response.metadata.get("job_id"),
        "agency": template.get("agency", "FDA / ICH"),
        "language": output_language,
    }


async def _fetch_trial_data(nct_id: str) -> dict:
    """Pull structured trial data from ClinicalTrials.gov for grounding."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"https://clinicaltrials.gov/api/v2/studies/{nct_id}",
                params={"format": "json"}
            )
            if r.status_code != 200:
                return {}
            p = r.json().get("protocolSection", {})
            id_mod = p.get("identificationModule", {})
            design_mod = p.get("designModule", {})
            status_mod = p.get("statusModule", {})
            desc_mod = p.get("descriptionModule", {})
            outcomes_mod = p.get("outcomesModule", {})
            eligibility_mod = p.get("eligibilityModule", {})

            primary_outcomes = [
                o.get("measure", "") for o in outcomes_mod.get("primaryOutcomes", [])
            ]
            secondary_outcomes = [
                o.get("measure", "") for o in outcomes_mod.get("secondaryOutcomes", [])[:5]
            ]

            return {
                "nct_id": nct_id,
                "title": id_mod.get("briefTitle", ""),
                "phase": design_mod.get("phases", []),
                "enrollment": design_mod.get("enrollmentInfo", {}).get("count", ""),
                "study_type": design_mod.get("studyType", ""),
                "allocation": design_mod.get("designInfo", {}).get("allocation", ""),
                "masking": design_mod.get("designInfo", {}).get("maskingInfo", {}).get("masking", ""),
                "primary_completion": status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
                "brief_summary": desc_mod.get("briefSummary", ""),
                "primary_outcomes": primary_outcomes,
                "secondary_outcomes": secondary_outcomes,
                "eligibility_criteria": eligibility_mod.get("eligibilityCriteria", "")[:1000],
                "min_age": eligibility_mod.get("minimumAge", ""),
                "sex": eligibility_mod.get("sex", ""),
            }
    except Exception as e:
        print(f"[TrialFetch] {e}")
        return {}


_LANG_INSTRUCTIONS = {
    "es": (
        "IDIOMA DE SALIDA: Español. Redacta todo el documento en español, "
        "usando terminología regulatoria formal latinoamericana. "
        "Mantén los nombres científicos (DCI) en latín/inglés según estándar farmacéutico internacional."
    ),
    "pt": (
        "IDIOMA DE SAÍDA: Português (Brasil). Redija todo o documento em português brasileiro, "
        "usando terminologia regulatória formal. "
        "Mantenha os nomes científicos (DCB/DCI) conforme padrão farmacêutico internacional."
    ),
    "en": "",
}


def _build_drafting_prompt(
    template: dict,
    drug_name: str,
    indication: str,
    knowledge_text: str,
    trial_data: dict,
    study_data: dict,
    section: Optional[str],
    language: str = "en",
) -> str:
    sections_list = "\n".join(f"  {s}" for s in template["sections"])

    trial_text = ""
    if trial_data:
        trial_text = f"""
TRIAL DATA (ClinicalTrials.gov {trial_data.get('nct_id', '')}):
  Title: {trial_data.get('title', '')}
  Phase: {', '.join(trial_data.get('phase', []))}
  Design: {trial_data.get('study_type', '')} | {trial_data.get('allocation', '')} | Masking: {trial_data.get('masking', '')}
  Enrollment: {trial_data.get('enrollment', '')} subjects
  Primary Completion: {trial_data.get('primary_completion', '')}
  Primary Outcomes: {'; '.join(trial_data.get('primary_outcomes', []))}
  Secondary Outcomes: {'; '.join(trial_data.get('secondary_outcomes', []))}
  Eligibility (excerpt): {trial_data.get('eligibility_criteria', '')[:500]}
  Summary: {trial_data.get('brief_summary', '')[:600]}
"""

    study_text = ""
    if study_data:
        study_text = f"\nADDITIONAL STUDY DATA:\n{study_data}"

    knowledge_section = f"\nREGULATORY KNOWLEDGE BASE:\n{knowledge_text}" if knowledge_text else ""

    if section:
        task = (
            f"Draft ONLY Section: {section}\n\n"
            f"This section is part of a {template['title']} for {drug_name} in {indication}. "
            f"Follow {template['guideline']} formatting requirements exactly."
        )
    else:
        task = (
            f"Draft a complete {template['title']} for the following compound and indication.\n\n"
            f"REQUIRED SECTIONS (per {template['guideline']}):\n{sections_list}\n\n"
            f"For each section: use proper ICH/FDA formatting, include placeholders [TO BE COMPLETED] "
            f"where site-specific data is needed, and flag any data gaps explicitly."
        )

    lang_instruction = _LANG_INSTRUCTIONS.get(language, "")
    agency = template.get("agency", "FDA / ICH")

    return f"""You are a senior regulatory affairs medical writer with 15+ years experience in {agency} submissions.
{lang_instruction}

TASK: {task}

DRUG: {drug_name}
INDICATION: {indication}
REGULATORY FRAMEWORK: {template['fda_ref']}
AGENCY: {agency}
{trial_text}{study_text}{knowledge_section}

INSTRUCTIONS:
- Follow {agency} formatting requirements precisely
- Use professional regulatory writing style (passive voice, precise terminology)
- Include all required section headers
- Mark data gaps as [DATA REQUIRED: description]
- Include statistical methodology placeholders where applicable
- Add appropriate safety language and disclaimers
- Do not fabricate specific efficacy numbers — use trial data provided or mark as [TO BE COMPLETED FROM STUDY DATA]
"""
