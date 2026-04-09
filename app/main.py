from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints.orchestration import router as orchestration_router
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Ariston AI LifeScience OS",
    description="Production-grade Agentic Orchestration for Life Sciences",
    version="2.0.0"
)

# CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse
import os

# Mount API Routers
app.include_router(orchestration_router, prefix="/api/v1", tags=["orchestration"])

# Mount React Production Build
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/assets", StaticFiles(directory=os.path.join(static_path, "assets")), name="assets")

@app.get("/{full_path:path}")
async def serve_react(full_path: str):
    # Serve assets if they exist, else serve index.html for SPA routing
    file_path = os.path.join(static_path, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(static_path, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
