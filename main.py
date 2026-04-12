from dotenv import load_dotenv
load_dotenv()

import os
os.makedirs("data", exist_ok=True)
os.makedirs("benchmarks", exist_ok=True)

from fastapi import FastAPI
from vinci_core.router import router as vinci_router
from ariston_pharma.router import router as pharma_router
from hippokron.router import router as hippokron_router
from darwina.router import router as darwina_router
from app.api.v1.endpoints.orchestration import router as orchestration_router
from app.api.v1.endpoints.latam import router as latam_router
from app.api.v1.endpoints.phase2 import router as phase2_router
from app.api.v1.endpoints.phase3 import router as phase3_router
from app.api.v1.endpoints.phase4 import router as phase4_router
from app.api.v1.endpoints.phase5 import router as phase5_router
from app.api.v1.endpoints.phase6 import router as phase6_router
from app.api.v1.endpoints.platform import router as platform_router
from app.localization.router import router as localization_router
from vinci_core.continuous_improvement.router import router as improvement_router
from app.pilot_programs.router import router as pilots_router
from vinci_core.swarm.router import router as swarm_router
from app.agents.router import router as agents_router
from vinci_core.rwe.router import router as rwe_router

app = FastAPI(
    title="Ariston AI — Life Sciences Platform",
    version="0.7.0",
    description=(
        "AI OS layer for Life Sciences. "
        "Phase 1: LATAM regulatory intelligence (ANVISA/COFEPRIS/INVIMA/ANMAT/ISP). "
        "Phase 2: Real-World Evidence, Pharmacovigilance, CSR generation. "
        "Phase 3: Drug discovery AI, biomarker discovery, clinical trial intelligence, FDA 510(k), international expansion. "
        "Phase 4: GxP audit trail, agent memory, multi-tenant RBAC, webhook events. "
        "Phase 5: Revenue infrastructure — billing, Stripe, SLA monitoring, customer portal. "
        "Phase 6: Data moat — LATAM connectors, semantic embeddings, RWE accumulation flywheel. "
        "Multi-agent swarm, composable pipelines, RAG-enriched responses."
    ),
)

# ── Core domain routers ────────────────────────────────────────────────────
app.include_router(vinci_router, prefix="/api/v1")
app.include_router(pharma_router, prefix="/api/v1")
app.include_router(hippokron_router, prefix="/api/v1")
app.include_router(darwina_router, prefix="/api/v1/darwina")

# ── Orchestration (background jobs + WebSocket + GxP audit) ───────────────
app.include_router(orchestration_router, prefix="/api/v1")

# ── Phase 1: LATAM Regulatory Intelligence ────────────────────────────────
app.include_router(latam_router, prefix="/api/v1")
app.include_router(localization_router, prefix="/api/v1")
app.include_router(pilots_router, prefix="/api/v1")
app.include_router(agents_router, prefix="/api/v1")
app.include_router(swarm_router, prefix="/api/v1")

# ── Phase 2: RWE + Pharmacovigilance + CSR ────────────────────────────────
app.include_router(phase2_router, prefix="/api/v1")
app.include_router(rwe_router, prefix="/api/v1")

# ── Phase 3: Drug Discovery + Clinical Trials + FDA 510(k) + International ─
app.include_router(phase3_router, prefix="/api/v1")

# ── Phase 4: GxP Audit + Agent Memory + RBAC + Webhooks ───────────────────
app.include_router(phase4_router, prefix="/api/v1")

# ── Phase 5: Revenue Infrastructure ───────────────────────────────────────
app.include_router(phase5_router, prefix="/api/v1")

# ── Phase 6: Data Moat — LATAM + Embeddings + RWE Accumulation ───────────
app.include_router(phase6_router, prefix="/api/v1")

# ── Platform Orchestration — Wire All Agents + Phases ─────────────────────
app.include_router(platform_router, prefix="/api/v1")

# ── Platform: Autonomous Continuous Improvement Loop ──────────────────────
app.include_router(improvement_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "platform": "Ariston AI",
        "version": "0.7.0",
        "roadmap": {
            "phase1": {
                "status": "active",
                "focus": "LATAM Go-to-Market",
                "capabilities": [
                    "regulatory_intelligence",
                    "latam_agencies_ANVISA_COFEPRIS_INVIMA_ANMAT_ISP",
                    "pilot_program_management",
                    "multi_agent_swarm",
                    "localization_es_pt",
                ],
            },
            "phase2": {
                "status": "active",
                "focus": "Real-World Evidence + Data Licensing",
                "capabilities": [
                    "pharmacovigilance_latam_cioms_medwatch",
                    "csr_generation_ich_e3",
                    "rwe_data_licensing",
                    "biomarker_discovery",
                ],
            },
            "phase3": {
                "status": "active",
                "focus": "Drug Discovery AI + International Expansion",
                "capabilities": [
                    "drug_target_identification",
                    "drug_repurposing_signals",
                    "clinical_trial_intelligence_latam",
                    "fda_510k_preparation_pccp",
                    "international_regulatory_ema_pmda_mhra_tga",
                    "rag_pubmed_opentargets_clinicaltrials",
                ],
            },
            "phase4": {
                "status": "active",
                "focus": "Platform Intelligence",
                "capabilities": [
                    "gxp_audit_trail_21cfr11_eu_annex11",
                    "agent_persistent_memory",
                    "multi_tenant_rbac",
                    "webhook_event_system",
                ],
            },
            "phase5": {
                "status": "active",
                "focus": "Revenue Infrastructure",
                "capabilities": [
                    "usage_metering_per_tenant",
                    "stripe_subscription_billing",
                    "sla_monitoring_p50_p95_uptime",
                    "customer_portal_dashboard",
                    "overage_billing_enforcement",
                ],
            },
            "phase6": {
                "status": "active",
                "focus": "Data Moat",
                "capabilities": [
                    "latam_data_connectors_DATASUS_SINAVE_SISPRO_SNVS_DEIS",
                    "semantic_embedding_search_cosine_similarity",
                    "rwe_accumulation_pipeline_freshness_monitor",
                    "disease_burden_paho_who_estimates",
                    "data_flywheel_rwe_to_biomarker_to_drug_discovery",
                ],
            },
        },
        "latam_agencies": ["ANVISA", "COFEPRIS", "INVIMA", "ANMAT", "ISP"],
        "international_agencies": ["EMA", "PMDA", "MHRA", "TGA", "Health Canada"],
        "rag_sources": ["PubMed", "Open Targets", "ClinicalTrials.gov", "FDA Labels"],
        "continuous_improvement": True,
        "gxp_compliant": True,
        "multi_tenant": True,
    }
