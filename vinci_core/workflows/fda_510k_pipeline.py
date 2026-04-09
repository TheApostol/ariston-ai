"""
FDA 510(k) Preparation Pipeline — Phase 3 / Ariston AI.

Generates 510(k) premarket notification documentation for AI-enabled
medical devices. Ariston's first 510(k) target: a clinical decision support
tool that meets the Cures Act CDS exemption criteria.

510(k) pathway (Phase 3 milestone per Execution Roadmap):
  - Median review time: 142 days
  - Cost: $150K–$500K
  - Strategy: find predicate device among 1,250+ cleared AI algorithms

This pipeline generates:
  1. Predicate Search   — identify matching cleared predicates
  2. Intended Use       — draft intended use statement (non-device if CDS exempt)
  3. Substantial Equiv  — argument for substantial equivalence
  4. Performance Data   — performance testing summary template
  5. PCCP Draft         — Predetermined Change Control Plan for adaptive AI
  6. Submission Package — assembled 510(k) shell

Regulatory strategy:
  - Phase 1: Non-device CDS (no 510k needed) — revenue immediately
  - Phase 3: First 510(k) for diagnostic aid — creates competitive moat
    (your cleared device becomes the predicate for competitors)
"""

from __future__ import annotations

from vinci_core.workflows.pipeline import Pipeline, PipelineContext, PipelineStep, step

_KNOWN_AI_PREDICATES = [
    {"k_number": "K222016", "device": "AI-Rad Companion (radiology AI)", "indication": "chest x-ray triage"},
    {"k_number": "K213872", "device": "ContaCT (stroke detection)", "indication": "large vessel occlusion"},
    {"k_number": "K190186", "device": "IDx-DR", "indication": "diabetic retinopathy screening"},
    {"k_number": "K213322", "device": "Viz.ai LVO", "indication": "large vessel occlusion"},
    {"k_number": "K221315", "device": "Aidoc bone-age", "indication": "pediatric bone age"},
    {"k_number": "K220551", "device": "Paige Prostate", "indication": "prostate cancer detection"},
    {"k_number": "K211015", "device": "DreaMed Advisor Pro", "indication": "insulin dose decision support"},
]


# ---------------------------------------------------------------------------
# Step 1: Predicate Search
# ---------------------------------------------------------------------------
@step("fda_predicate_search")
async def predicate_search_step(ctx: PipelineContext) -> PipelineContext:
    """Identify cleared AI predicates matching the intended device."""
    device_data = ctx.metadata.get("device_data") or {}
    indication = device_data.get("indication", "").lower()
    device_type = device_data.get("device_type", "").lower()

    # Simple keyword match — production: query FDA 510k database API
    matches = [
        p for p in _KNOWN_AI_PREDICATES
        if any(kw in p["indication"] for kw in indication.split()[:3])
        or any(kw in p["device"].lower() for kw in device_type.split()[:2])
    ]

    if not matches:
        matches = _KNOWN_AI_PREDICATES[:2]  # fallback — show 2 examples

    ctx.results["predicate_candidates"] = matches
    ctx.results["indication"] = indication or device_data.get("indication", "clinical decision support")
    return ctx


# ---------------------------------------------------------------------------
# Step 2: Intended Use Statement
# ---------------------------------------------------------------------------
@step("fda_intended_use")
async def intended_use_step(ctx: PipelineContext) -> PipelineContext:
    """Draft the intended use and indications for use statements."""
    from vinci_core.engine import engine

    device_data = ctx.metadata.get("device_data") or {}
    indication  = ctx.results.get("indication", "clinical decision support")

    prompt = (
        f"Draft an FDA 510(k) Intended Use and Indications for Use statement for:\n"
        f"Device: {device_data.get('device_name', 'AI Clinical Decision Support Software')}\n"
        f"Indication: {indication}\n"
        f"Device type: Software as a Medical Device (SaMD)\n\n"
        f"Requirements:\n"
        f"1. Intended Use: what the device does (machine function)\n"
        f"2. Indications for Use: clinical context of use\n"
        f"3. Contraindications if any\n"
        f"4. Assess whether the Cures Act CDS exemption might apply (4 criteria)\n"
        f"5. Note if De Novo vs 510(k) is more appropriate\n"
        f"Use precise FDA regulatory language. Avoid overclaiming."
    )

    response = await engine.run(prompt=prompt, layer="pharma", use_rag=False)
    ctx.results["intended_use_draft"] = response.content
    return ctx


# ---------------------------------------------------------------------------
# Step 3: Substantial Equivalence Argument
# ---------------------------------------------------------------------------
@step("fda_substantial_equivalence")
async def substantial_equivalence_step(ctx: PipelineContext) -> PipelineContext:
    """Draft the substantial equivalence (SE) argument."""
    predicates = ctx.results.get("predicate_candidates", [])
    indication  = ctx.results.get("indication", "")

    if not predicates:
        ctx.results["se_argument"] = "[No predicate identified — De Novo pathway recommended]"
        return ctx

    predicate = predicates[0]
    se_text = (
        f"SUBSTANTIAL EQUIVALENCE ARGUMENT\n\n"
        f"Predicate Device: {predicate['device']} ({predicate['k_number']})\n"
        f"Predicate Indication: {predicate['indication']}\n\n"
        f"Comparison:\n"
        f"1. SAME INTENDED USE: Both devices are intended to [align with predicate indication].\n"
        f"   Our device: {indication}\n"
        f"   Predicate: {predicate['indication']}\n\n"
        f"2. SAME TECHNOLOGICAL CHARACTERISTICS: Both use deep learning / AI algorithms "
        f"to analyze clinical data and output decision support.\n\n"
        f"3. PERFORMANCE: Our device achieves comparable or better performance on the "
        f"same device type metrics (sensitivity, specificity, AUC).\n"
        f"   [Insert performance comparison table — clinical data required]\n\n"
        f"4. CONCLUSION: The subject device is substantially equivalent to "
        f"{predicate['device']} ({predicate['k_number']}) pursuant to section 513(i) "
        f"of the Federal Food, Drug, and Cosmetic Act.\n\n"
        f"*Note: This is a draft SE argument. Clinical performance data and detailed "
        f"technical comparison required before submission.*"
    )

    ctx.results["se_argument"] = se_text
    return ctx


# ---------------------------------------------------------------------------
# Step 4: PCCP Draft (Predetermined Change Control Plan)
# ---------------------------------------------------------------------------
@step("fda_pccp")
async def pccp_step(ctx: PipelineContext) -> PipelineContext:
    """Draft the Predetermined Change Control Plan for adaptive AI."""
    pccp = (
        "PREDETERMINED CHANGE CONTROL PLAN (PCCP)\n"
        "Per FDA Guidance: Marketing Submission Recommendations for a PCCP\n\n"
        "1. DESCRIPTION OF MODIFICATION PROTOCOL\n"
        "   Permitted modifications include:\n"
        "   a) Model retraining on expanded datasets (same intended use)\n"
        "   b) Performance threshold adjustments within ±5% of cleared thresholds\n"
        "   c) Addition of new data input modalities (requires supplemental 510k)\n\n"
        "2. IMPACT ASSESSMENT PROTOCOL\n"
        "   Each modification must be assessed against:\n"
        "   - Clinical safety impact (AUC, sensitivity, specificity)\n"
        "   - Bias assessment across demographic subgroups\n"
        "   - Distribution shift detection (concept drift)\n"
        "   Threshold for supplemental 510k: >5% change in primary performance metric\n\n"
        "3. PERFORMANCE MONITORING PROTOCOL\n"
        "   Post-deployment monitoring via:\n"
        "   - Continuous AUC tracking (weekly)\n"
        "   - Adverse event / near-miss logging\n"
        "   - Quarterly performance reports to FDA per PMS plan\n\n"
        "4. METHODOLOGY\n"
        "   Algorithm: [Describe architecture — CNN/Transformer/ensemble]\n"
        "   Training data: [De-identified, IRB-approved, LATAM-inclusive datasets]\n"
        "   Validation: [Prospective, held-out test set, external validation cohort]\n\n"
        "*Note: PCCP must be finalized with device-specific performance data "
        "and reviewed by regulatory counsel before FDA submission.*"
    )

    ctx.results["pccp_draft"] = pccp
    return ctx


# ---------------------------------------------------------------------------
# Step 5: Assemble Submission Package Shell
# ---------------------------------------------------------------------------
@step("fda_assemble_package")
async def assemble_package_step(ctx: PipelineContext) -> PipelineContext:
    """Assemble the 510(k) submission package shell."""
    predicates = ctx.results.get("predicate_candidates", [])
    predicate_str = ", ".join(f"{p['k_number']} ({p['device']})" for p in predicates[:2])

    package = f"""
FDA 510(k) PREMARKET NOTIFICATION — DRAFT SHELL
Generated by Ariston AI | Phase 3 Regulatory Module
{'='*60}

SECTION 1: SUBMITTER INFORMATION
[Insert sponsor/company details]

SECTION 2: DEVICE NAME AND CLASSIFICATION
Device Name: [Insert]
Classification: Class II — Software as a Medical Device (SaMD)
Product Code: [Identify from FDA product codes — e.g., QMF for AI radiology]
Regulation: 21 CFR 892.2050 (or applicable)

SECTION 3: PREDICATE DEVICE(S)
{predicate_str or '[Predicate to be identified via 510k database search]'}

SECTION 4: DEVICE DESCRIPTION
[Insert functional description, hardware/software specifications]

SECTION 5: INTENDED USE AND INDICATIONS FOR USE
{ctx.results.get('intended_use_draft', '[Draft required — see Phase 3 pipeline output]')}

SECTION 6: SUBSTANTIAL EQUIVALENCE DISCUSSION
{ctx.results.get('se_argument', '[SE argument required]')}

SECTION 7: PERFORMANCE DATA
7.1 Analytical Studies: [Insert bench testing]
7.2 Clinical Studies: [Insert clinical validation study]
7.3 Software Documentation: Per FDA Software as a Medical Device guidance
7.4 Cybersecurity: [Insert per FDA cybersecurity guidance]

SECTION 8: PREDETERMINED CHANGE CONTROL PLAN
{ctx.results.get('pccp_draft', '[PCCP required for adaptive AI devices]')}

SECTION 9: LABELING
[Insert IFU, device labels per 21 CFR Part 801]

SECTION 10: DECLARATIONS
[Insert truthfulness declarations per 21 CFR 807.87]

{'='*60}
STATUS: DRAFT SHELL — All [insert] sections require clinical/technical data
REGULATORY STRATEGY: Target CDS exemption for Phase 1; 510(k) for Phase 3 product
ESTIMATED TIMELINE: 142 days median review after complete submission
ESTIMATED COST: $150K–$500K submission preparation
{'='*60}
"""

    ctx.final_content = package.strip()
    return ctx


# ---------------------------------------------------------------------------
# Assembled Pipeline
# ---------------------------------------------------------------------------
fda_510k_pipeline = Pipeline(
    steps=[
        predicate_search_step,
        intended_use_step,
        substantial_equivalence_step,
        pccp_step,
        assemble_package_step,
    ],
    name="fda_510k",
)
