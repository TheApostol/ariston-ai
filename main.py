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
from app.localization.router import router as localization_router
from vinci_core.continuous_improvement.router import router as improvement_router
from app.pilot_programs.router import router as pilots_router
from vinci_core.swarm.router import router as swarm_router
from app.agents.router import router as agents_router
from vinci_core.rwe.router import router as rwe_router

app = FastAPI(
    title="Ariston AI — Life Sciences Platform",
    version="0.5.0",
    description=(
        "AI OS layer for Life Sciences. "
        "Phase 1: LATAM regulatory intelligence (ANVISA/COFEPRIS/INVIMA/ANMAT/ISP). "
        "Phase 2: Real-World Evidence, Pharmacovigilance, CSR generation. "
        "Phase 3: Biomarker discovery, drug discovery AI. "
        "Multi-agent swarm, composable pipelines, GxP audit trail."
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

# ── Phase 3: Clinical Trial Intelligence + FDA 510(k) ─────────────────────
app.include_router(phase3_router, prefix="/api/v1")

# ── Platform: Autonomous Continuous Improvement Loop ──────────────────────
app.include_router(improvement_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "platform": "Ariston AI",
        "version": "0.5.0",
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
                "status": "building",
                "focus": "Real-World Evidence + Data Licensing",
                "capabilities": [
                    "pharmacovigilance_latam",
                    "csr_generation",
                    "rwe_data_licensing",
                    "biomarker_discovery_preview",
                ],
            },
            "phase3": {
                "status": "building",
                "focus": "Drug Discovery AI + FDA 510k",
                "capabilities": [
                    "biomarker_discovery",
                    "clinical_trial_intelligence_latam",
                    "clinical_decision_support",
                    "fda_510k_preparation",
                    "pccp_adaptive_ai",
                    "international_expansion",
                ],
            },
        },
        "latam_agencies": ["ANVISA", "COFEPRIS", "INVIMA", "ANMAT", "ISP"],
        "continuous_improvement": True,
    }
