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

app = FastAPI(
    title="Ariston AI — Life Sciences Platform",
    version="0.3.0",
    description=(
        "AI orchestration engine for pharmaceutical regulatory intelligence, "
        "clinical trial optimization, real-world evidence analysis, "
        "and LATAM market regulatory submissions (ANVISA, COFEPRIS, INVIMA, ANMAT, ISP)."
    ),
)

# Domain-specific routers
app.include_router(vinci_router, prefix="/api/v1")
app.include_router(pharma_router, prefix="/api/v1")
app.include_router(hippokron_router, prefix="/api/v1")
app.include_router(darwina_router, prefix="/api/v1/darwina")

# Primary orchestration router (background jobs + WebSocket + audit)
app.include_router(orchestration_router, prefix="/api/v1")

# LATAM Regulatory Intelligence (Phase 1 go-to-market)
app.include_router(latam_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "platform": "Ariston AI",
        "version": "0.3.0",
        "markets": ["global", "latam"],
        "latam_agencies": ["ANVISA", "COFEPRIS", "INVIMA", "ANMAT", "ISP"],
    }
