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
from app.localization.router import router as localization_router
from vinci_core.continuous_improvement.router import router as improvement_router
from app.pilot_programs.router import router as pilots_router
from vinci_core.swarm.router import router as swarm_router
from app.agents.router import router as agents_router

app = FastAPI(
    title="Ariston AI — Life Sciences Platform",
    version="0.4.0",
    description=(
        "AI orchestration engine for pharmaceutical regulatory intelligence, "
        "clinical trial optimization, real-world evidence analysis, "
        "and LatAm market launch. Powered by a multi-agent swarm with "
        "composable pipelines and autonomous improvement loops."
    ),
)

# Domain-specific routers
app.include_router(vinci_router, prefix="/api/v1")
app.include_router(pharma_router, prefix="/api/v1")
app.include_router(hippokron_router, prefix="/api/v1")
app.include_router(darwina_router, prefix="/api/v1/darwina")

# Primary orchestration router (background jobs + WebSocket + audit)
app.include_router(orchestration_router, prefix="/api/v1")

# LatAm Localization Layer
app.include_router(localization_router, prefix="/api/v1")

# Autonomous Continuous Improvement Loop
app.include_router(improvement_router, prefix="/api/v1")

# LatAm Pilot Program Framework
app.include_router(pilots_router, prefix="/api/v1")

# Multi-Agent Swarm Orchestrator
app.include_router(swarm_router, prefix="/api/v1")

# Individual Agent REST API
app.include_router(agents_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "platform": "Ariston AI", "version": "0.4.0"}
