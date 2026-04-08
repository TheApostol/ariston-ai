from fastapi import FastAPI
from config import settings
from vinci_core.router import router as vinci_router
from hippokron.router import router as hippokron_router
from darwina.router import router as darwina_router
from ariston_pharma.router import router as pharma_router

app = FastAPI(
    title=settings.app_name,
    description="Multi-layer AI platform: Vinci Core · HippoKron · Darwina · Ariston Pharma",
    version="0.1.0",
)

app.include_router(vinci_router, prefix="/api/v1")
app.include_router(hippokron_router, prefix="/api/v1")
app.include_router(darwina_router, prefix="/api/v1")
app.include_router(pharma_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "platform": settings.app_name,
        "layers": ["vinci_core", "hippokron", "darwina", "ariston_pharma"],
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
