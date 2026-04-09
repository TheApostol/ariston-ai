"""
Clinical Study Report (CSR) Pipeline — Phase 2 / Ariston AI LATAM.

Generates ICH E3-compliant Clinical Study Reports for regulatory submission.
This is the primary revenue-generating use case in the Execution Roadmap:
  "A Merck/McKinsey pilot demonstrated that generative AI cuts CSR writing
   time from 180 to 80 hours — a 55% reduction."

CSR sections per ICH E3 guideline:
  1. Title page + synopsis
  2. Introduction
  3. Study objectives
  4. Investigational plan
  5. Study subjects
  6. Efficacy evaluation
  7. Safety evaluation
  8. Discussion + conclusions
  9. References + appendices

LATAM adaptations:
  - ANVISA: Module 5.3.5 CTD format required
  - COFEPRIS: Spanish translation + NOM-compliant terminology
  - INVIMA: Decreto 677/1995 dossier alignment
  - ANMAT: Decreto 150/92 reference product comparison
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from vinci_core.workflows.pipeline import Pipeline, PipelineContext, PipelineStep, step


# ---------------------------------------------------------------------------
# Step 1: Protocol Parsing
# ---------------------------------------------------------------------------
@step("csr_protocol_parse")
async def protocol_parse_step(ctx: PipelineContext) -> PipelineContext:
    """Extract study metadata from protocol context."""
    study = ctx.metadata.get("study_data") or {}

    ctx.results["study_title"]    = study.get("title", "Phase III Clinical Study")
    ctx.results["study_number"]   = study.get("number", "STUDY-001")
    ctx.results["drug_name"]      = study.get("drug_name", "Investigational Product")
    ctx.results["indication"]     = study.get("indication", "Not specified")
    ctx.results["phase"]          = study.get("phase", "III")
    ctx.results["sponsor"]        = study.get("sponsor", "Sponsor Organization")
    ctx.results["study_period"]   = study.get("study_period", "Not specified")
    ctx.results["subject_count"]  = study.get("subject_count", 0)
    ctx.results["primary_endpoint"] = study.get("primary_endpoint", "Not specified")
    ctx.results["latam_countries"] = study.get("countries", ["brazil", "mexico"])

    return ctx


# ---------------------------------------------------------------------------
# Step 2: Synopsis Generation (AI-assisted)
# ---------------------------------------------------------------------------
@step("csr_synopsis")
async def synopsis_step(ctx: PipelineContext) -> PipelineContext:
    """Generate the ICH E3 synopsis using the engine."""
    from vinci_core.engine import engine

    synopsis_prompt = (
        f"Generate an ICH E3-compliant clinical study synopsis for:\n\n"
        f"Study: {ctx.results['study_title']} ({ctx.results['study_number']})\n"
        f"Drug: {ctx.results['drug_name']}\n"
        f"Indication: {ctx.results['indication']}\n"
        f"Phase: {ctx.results['phase']}\n"
        f"Sponsor: {ctx.results['sponsor']}\n"
        f"Study Period: {ctx.results['study_period']}\n"
        f"Subjects: {ctx.results['subject_count']}\n"
        f"Primary Endpoint: {ctx.results['primary_endpoint']}\n"
        f"LATAM Countries: {', '.join(c.upper() for c in ctx.results['latam_countries'])}\n\n"
        f"Structure the synopsis per ICH E3 Section 2: include objectives, design, "
        f"subject disposition, efficacy results (placeholder), safety summary, "
        f"and conclusion. Flag fields requiring clinical data insertion."
    )

    response = await engine.run(prompt=synopsis_prompt, layer="pharma", use_rag=False)
    ctx.results["synopsis"] = response.content
    return ctx


# ---------------------------------------------------------------------------
# Step 3: Safety Section (AE Summary Table)
# ---------------------------------------------------------------------------
@step("csr_safety_section")
async def safety_section_step(ctx: PipelineContext) -> PipelineContext:
    """Generate the safety evaluation section (ICH E3 Section 12)."""
    study = ctx.metadata.get("study_data") or {}
    ae_summary = study.get("ae_summary", {})

    safety_text = (
        f"## 12. SAFETY EVALUATION\n\n"
        f"### 12.1 Extent of Exposure\n"
        f"A total of {ctx.results.get('subject_count', '[N]')} subjects received "
        f"{ctx.results.get('drug_name', 'investigational product')}.\n\n"
        f"### 12.2 Adverse Events\n"
        f"Total AEs reported: {ae_summary.get('total_ae', '[N — insert from dataset]')}\n"
        f"Serious AEs (SAEs): {ae_summary.get('total_sae', '[N — insert from dataset]')}\n"
        f"AEs leading to discontinuation: {ae_summary.get('discontinuations', '[N — insert from dataset]')}\n"
        f"Deaths: {ae_summary.get('deaths', '[N — insert from dataset]')}\n\n"
        f"### 12.3 Deaths, SAEs, and Other Significant AEs\n"
        f"[Clinical data required — insert MedDRA-coded SAE narratives]\n\n"
        f"### 12.4 Clinical Laboratory Evaluations\n"
        f"[Laboratory data tables required — insert from EDC system]\n\n"
        f"### 12.5 Vital Signs, Physical Findings, and Other Observations\n"
        f"[Clinical observation data required]\n\n"
        f"*Note: Sections marked [insert] require data from the clinical database "
        f"prior to regulatory submission.*"
    )

    ctx.results["safety_section"] = safety_text
    return ctx


# ---------------------------------------------------------------------------
# Step 4: LATAM Regulatory Adaptation
# ---------------------------------------------------------------------------
@step("csr_latam_adaptation")
async def latam_adaptation_step(ctx: PipelineContext) -> PipelineContext:
    """Add LATAM-specific CTD module references and agency annotations."""
    countries = ctx.results.get("latam_countries", [])

    adaptations = {
        "brazil": (
            "ANVISA CTD Module 5.3.5: This CSR is formatted per ICH E3 and "
            "ready for inclusion in the eCTD Module 5 dossier. Portuguese translation "
            "of the synopsis required for SOLICITA submission."
        ),
        "mexico": (
            "COFEPRIS: CSR must accompany NOM-compliant study protocol. "
            "Spanish translation required. Reference product comparison "
            "per NOM-177-SSA1 bioequivalence standards if applicable."
        ),
        "colombia": (
            "INVIMA Decreto 677/1995: CSR included in dossier Sección V. "
            "SIVICOS electronic submission. Spanish translation required."
        ),
        "argentina": (
            "ANMAT Decreto 150/92: CSR attached to Módulo 5 CTD. "
            "SAID electronic system. Local Director Técnico co-signature required."
        ),
        "chile": (
            "ISP DS 3/2010: CSR as supporting document for registro sanitario. "
            "Spanish translation required. ISP digital portal submission."
        ),
    }

    latam_notes = "\n\n## LATAM REGULATORY ADAPTATION NOTES\n"
    for country in countries:
        note = adaptations.get(country.lower())
        if note:
            latam_notes += f"\n**{country.upper()}:** {note}\n"

    ctx.results["latam_adaptation_notes"] = latam_notes
    return ctx


# ---------------------------------------------------------------------------
# Step 5: CSR Assembly
# ---------------------------------------------------------------------------
@step("csr_assemble")
async def assemble_step(ctx: PipelineContext) -> PipelineContext:
    """Assemble the full CSR document from all generated sections."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    csr = f"""
CLINICAL STUDY REPORT
ICH E3 FORMAT — ARISTON AI GENERATED DRAFT
{'='*60}

TITLE: {ctx.results.get('study_title')}
STUDY NUMBER: {ctx.results.get('study_number')}
DRUG: {ctx.results.get('drug_name')}
SPONSOR: {ctx.results.get('sponsor')}
INDICATION: {ctx.results.get('indication')}
PHASE: {ctx.results.get('phase')}
GENERATED: {now}
STATUS: DRAFT — Requires clinical data insertion before submission

{'='*60}

## 1. TITLE PAGE
[Complete per sponsor standard operating procedures]

## 2. SYNOPSIS
{ctx.results.get('synopsis', '[Synopsis generation pending]')}

## 3. TABLE OF CONTENTS
[Auto-generated on final document compilation]

## 4. LIST OF ABBREVIATIONS
[Insert standard ICH E3 + study-specific abbreviations]

## 5. ETHICS
[Insert IRB/IEC approval details and informed consent documentation]

## 6. INVESTIGATORS AND STUDY ADMINISTRATIVE STRUCTURE
[Insert investigator list — LATAM sites: {', '.join(c.upper() for c in ctx.results.get('latam_countries', []))}]

## 7. INTRODUCTION
[Insert therapeutic area background and rationale for study]

## 8. STUDY OBJECTIVES
Primary: {ctx.results.get('primary_endpoint')}
[Insert secondary and exploratory endpoints]

## 9. INVESTIGATIONAL PLAN
[Insert study design, randomization, blinding, treatment arms]

## 10. STUDY SUBJECTS
Total enrolled: {ctx.results.get('subject_count')}
[Insert disposition table, inclusion/exclusion criteria]

## 11. EFFICACY EVALUATION
[Insert primary and secondary endpoint analysis — statistical data required]

{ctx.results.get('safety_section', '')}

## 13. DISCUSSION AND OVERALL CONCLUSIONS
[Insert benefit-risk assessment and clinical conclusions]

## 14. REFERENCES
[Insert bibliography]

## 15. APPENDICES
[Insert protocol, amendments, CRF samples, statistical analysis plan]

{ctx.results.get('latam_adaptation_notes', '')}

{'='*60}
DOCUMENT INTEGRITY
Generated by: Ariston AI CSR Pipeline
ICH E3 Compliance: Structural template only — data insertion required
GxP Note: This draft requires medical review and sponsor QA before submission.
{'='*60}
"""

    ctx.final_content = csr.strip()
    return ctx


# ---------------------------------------------------------------------------
# Assembled Pipeline
# ---------------------------------------------------------------------------
csr_pipeline = Pipeline(
    steps=[
        protocol_parse_step,
        synopsis_step,
        safety_section_step,
        latam_adaptation_step,
        assemble_step,
    ],
    name="csr_latam",
)
