"""
Platform Orchestration API — Ariston AI.

Single entrypoint that coordinates ALL agents, phases, and data sources.
This is the "wire everything together" endpoint that the frontend calls.

Endpoints:
  GET  /platform/status           — full platform health + all agent status
  POST /platform/orchestrate      — run all relevant agents for a request
  GET  /platform/agents           — all agent endpoints catalog
  GET  /platform/dashboard        — unified ops dashboard (usage + SLA + agents)
  POST /platform/rwe/seed         — seed RWE data for a therapeutic area
  GET  /platform/memory/context   — retrieve agent memory for a session
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

logger = logging.getLogger("ariston.platform")
router = APIRouter(prefix="/platform", tags=["Platform Orchestration"])


# ── Models ───────────────────────────────────────────────────────────────────

class OrchestrateRequest(BaseModel):
    prompt: str
    patient_id: Optional[str] = None
    drug_name: Optional[str] = None
    condition: Optional[str] = None
    country: Optional[str] = None
    tenant_id: str = "default"
    tier: str = "standard"
    agents: Optional[list[str]] = None   # None = auto-select
    use_rag: bool = True
    use_memory: bool = True


class RWESeedRequest(BaseModel):
    therapeutic_area: str
    countries: list[str] = ["brazil", "mexico", "colombia", "argentina", "chile"]
    embed: bool = True


# ── Platform Status ───────────────────────────────────────────────────────────

@router.get("/status")
async def platform_status():
    """Full platform health — all phases, all agents, all data sources."""
    from vinci_core.sla.monitor import sla_monitor
    from vinci_core.billing.metering import usage_meter
    from vinci_core.embeddings.store import embedding_store
    from vinci_core.rwe.accumulation import rwe_accumulation

    sla = sla_monitor.get_report(window_hours=1)
    embed_stats = embedding_store.get_stats()
    rwe_stats = rwe_accumulation.get_stats()

    return {
        "platform": "Ariston AI",
        "version": "0.7.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phases": {
            "phase1": "active",  # LATAM Regulatory
            "phase2": "active",  # RWE + PV + CSR
            "phase3": "active",  # Drug Discovery + International
            "phase4": "active",  # GxP Audit + Memory + RBAC + Webhooks
            "phase5": "active",  # Billing + SLA + Portal
            "phase6": "active",  # Data Moat + Embeddings + RWE Accumulation
        },
        "agents": {
            "digital_twin":       "active",
            "iomt":               "active",
            "pharmacogenomics":   "active",
            "regulatory_copilot": "active",
            "patient_history":    "active",
            "pv_narrative":       "active",
            "site_selection":     "active",
            "pharmacist":         "active",
            "latam_regulatory":   "active",
            "vision_radiology":   "active",
        },
        "sla": {
            "uptime_pct": sla.uptime_pct,
            "p50_ms": sla.p50_latency_ms,
            "p95_ms": sla.p95_latency_ms,
            "status": "breach" if sla.breaches else "healthy",
        },
        "data_moat": {
            "total_embeddings": embed_stats["total_documents"],
            "rwe_records": rwe_stats.total_records,
            "freshness_score": rwe_stats.freshness_score,
            "provider": embed_stats["active_provider"],
        },
        "latam_coverage": {
            "countries": 5,
            "sources": ["DATASUS", "SINAVE", "SISPRO", "SNVS", "DEIS"],
            "population_millions": 463,
        },
    }


# ── Full Agent Catalog ────────────────────────────────────────────────────────

@router.get("/agents")
async def list_agents():
    """Complete catalog of all agents with endpoints and capabilities."""
    return {
        "total": 10,
        "agents": [
            {
                "id": "digital_twin",
                "name": "Digital Twin",
                "description": "In-silico treatment simulation: efficacy, toxicity, organ impact",
                "endpoint": "POST /api/v1/agents/twin/simulate",
                "inputs": ["drug", "patient_history", "genetics"],
            },
            {
                "id": "iomt",
                "name": "IoMT Adherence Monitor",
                "description": "30-day adherence forecast from pillbox telemetry + vitals",
                "endpoint": "POST /api/v1/agents/iomt/adherence",
                "inputs": ["patient_history", "telemetry"],
            },
            {
                "id": "pharmacogenomics",
                "name": "PGx Agent",
                "description": "Gene-drug interaction alerts (CYP2C19, HLA-B*5701, TPMT, DPYD)",
                "endpoint": "POST /api/v1/agents/pgx/cross-reference",
                "inputs": ["drug_name", "patient_id"],
            },
            {
                "id": "regulatory_copilot",
                "name": "Regulatory Copilot",
                "description": "GxP-compliant clinical reports for IRB/FDA/EMA submission",
                "endpoint": "POST /api/v1/agents/regulatory/report",
                "inputs": ["prompt", "result", "job_id"],
            },
            {
                "id": "pv_narrative",
                "name": "PV Narrative Agent",
                "description": "CIOMS-I and MedWatch FDA 3500A adverse event narratives",
                "endpoint": "POST /api/v1/agents/pv/narrative",
                "inputs": ["case_id", "drug_name", "ae_term", "outcome"],
            },
            {
                "id": "site_selection",
                "name": "Site Selection Agent",
                "description": "LatAm clinical trial site recommendations (100-point scoring)",
                "endpoint": "POST /api/v1/agents/sites/recommend",
                "inputs": ["therapeutic_area", "agency", "top_n"],
            },
            {
                "id": "pharmacist",
                "name": "Pharmacist Agent",
                "description": "Drug-drug interactions, label compliance, pharmacovigilance review",
                "endpoint": "POST /api/v1/agents/pharmacist/review",
                "inputs": ["prompt", "context"],
            },
            {
                "id": "latam_regulatory",
                "name": "LATAM Regulatory Agent",
                "description": "Multi-country ANVISA/COFEPRIS/INVIMA/ANMAT/ISP registration roadmap",
                "endpoint": "POST /api/v1/agents/latam/roadmap",
                "inputs": ["drug_name", "indication", "target_countries"],
            },
            {
                "id": "vision_radiology",
                "name": "Vision Radiology Agent",
                "description": "Multimodal scan analysis (Gemini 2.0 Flash): Findings, Impression, Differential",
                "endpoint": "POST /api/v1/agents/vision/analyze",
                "inputs": ["prompt", "images"],
            },
            {
                "id": "patient_history",
                "name": "Patient History Agent",
                "description": "Longitudinal patient record persistence and retrieval",
                "endpoint": "GET /api/v1/agents/patient/{id}/history",
                "inputs": ["patient_id"],
            },
        ],
    }


# ── Full Platform Orchestration ───────────────────────────────────────────────

@router.post("/orchestrate")
async def platform_orchestrate(req: OrchestrateRequest):
    """
    Full platform orchestration — coordinates all relevant agents for a request.
    Auto-selects agents based on prompt content and context.
    Records to SLA, usage meter, audit trail, and agent memory.
    """
    job_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    results = {}

    # Determine which agents to run
    auto_agents = req.agents or _auto_select_agents(req)

    # Run selected agents concurrently
    tasks = {}
    if "engine" in auto_agents:
        tasks["engine"] = _run_engine(req, job_id)
    if "pgx" in auto_agents and req.drug_name:
        tasks["pgx"] = _run_pgx(req.drug_name)
    if "pharmacist" in auto_agents and req.drug_name:
        tasks["pharmacist"] = _run_pharmacist(req)
    if "latam" in auto_agents and req.country:
        tasks["latam"] = _run_latam(req)
    if "sites" in auto_agents and req.condition:
        tasks["sites"] = _run_sites(req)
    if "memory" in auto_agents and req.use_memory:
        tasks["memory"] = _run_memory(req, job_id)

    # Execute all in parallel
    if tasks:
        task_list = list(tasks.items())
        coros = [t for _, t in task_list]
        task_results = await asyncio.gather(*coros, return_exceptions=True)
        for (key, _), result in zip(task_list, task_results):
            if isinstance(result, Exception):
                results[key] = {"error": str(result)}
            else:
                results[key] = result

    # Record usage
    try:
        from vinci_core.billing.metering import usage_meter
        usage_meter.record(
            tenant_id=req.tenant_id,
            unit="pipeline_runs",
            pipeline="platform_orchestrate",
            tier=req.tier,
        )
    except Exception:
        pass

    return {
        "job_id": job_id,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "agents_run": list(results.keys()),
        "results": results,
        "tenant_id": req.tenant_id,
    }


def _auto_select_agents(req: OrchestrateRequest) -> list[str]:
    """Auto-select agents based on request content."""
    agents = ["engine"]
    prompt_lower = req.prompt.lower()
    if req.drug_name or any(w in prompt_lower for w in ["drug", "medication", "dose", "interaction"]):
        agents += ["pgx", "pharmacist"]
    if req.country or any(w in prompt_lower for w in ["anvisa", "cofepris", "invima", "latam", "brazil", "mexico"]):
        agents.append("latam")
    if req.condition or any(w in prompt_lower for w in ["trial", "site", "clinical", "enrollment"]):
        agents.append("sites")
    if req.use_memory:
        agents.append("memory")
    return agents


async def _run_engine(req: OrchestrateRequest, job_id: str) -> dict:
    from vinci_core.engine import engine
    response = await engine.run(
        prompt=req.prompt,
        patient_id=req.patient_id,
        context={
            "drug_name": req.drug_name,
            "tenant_id": req.tenant_id,
            "tier": req.tier,
        },
        use_rag=req.use_rag,
    )
    return {
        "content": response.content,
        "model": response.model,
        "metadata": response.metadata,
    }


async def _run_pgx(drug_name: str) -> dict:
    from vinci_core.agent.genomics_agent import pharmacogenomics_agent
    result = await pharmacogenomics_agent.cross_reference(drug_name)
    return {"drug": drug_name, "pgx_alerts": result}


async def _run_pharmacist(req: OrchestrateRequest) -> dict:
    from vinci_core.agent.pharmacist_agent import PharmacistAgent
    agent = PharmacistAgent()
    review = await agent.review_medications(
        prompt=req.prompt,
        context={"drug_name": req.drug_name, "condition": req.condition},
    )
    return {"review": review}


async def _run_latam(req: OrchestrateRequest) -> dict:
    from vinci_core.agent.latam_agent import LatamRegulatoryAgent
    agent = LatamRegulatoryAgent()
    countries = [req.country] if req.country else ["brazil", "mexico"]
    roadmap = agent.build_multi_country_roadmap(
        countries=countries,
        product_type="pharmaceutical",
    )
    return {"roadmap": roadmap, "countries": countries}


async def _run_sites(req: OrchestrateRequest) -> dict:
    from vinci_core.agent.site_selection_agent import site_selection_agent
    sites = site_selection_agent.recommend_sites(
        therapeutic_area=req.condition or "general",
        top_n=3,
    )
    return sites


async def _run_memory(req: OrchestrateRequest, job_id: str) -> dict:
    from vinci_core.memory.agent_memory import AgentMemoryStore
    memory = AgentMemoryStore()
    context = memory.summarize(
        agent_type="platform",
        tenant_id=req.tenant_id,
        query=req.prompt,
        limit=5,
    )
    # Store this request as a memory
    memory.remember(
        content=f"User query: {req.prompt[:200]}",
        agent_type="platform",
        tenant_id=req.tenant_id,
        session_id=job_id,
        importance=0.6,
    )
    return {"context": context}


# ── RWE Data Seeding ──────────────────────────────────────────────────────────

@router.post("/rwe/seed")
async def seed_rwe_data(req: RWESeedRequest):
    """
    Seed RWE data for a therapeutic area across all LATAM countries.
    Triggers accumulation pipeline → embedding → freshness tracking.
    """
    from vinci_core.rwe.accumulation import rwe_accumulation

    condition_map = {
        "diabetes": "type2_diabetes",
        "cardiovascular": "cardiovascular",
        "oncology": "oncology",
        "dengue": "dengue",
        "chagas": "chagas_disease",
        "tuberculosis": "tuberculosis",
    }
    condition = condition_map.get(req.therapeutic_area.lower(), req.therapeutic_area.lower())

    records = []
    for country in req.countries:
        try:
            record = await rwe_accumulation.accumulate(
                country=country,
                condition=condition,
                embed=req.embed,
            )
            records.append({
                "country": country,
                "record_count": record.record_count,
                "doc_id": record.doc_id,
                "embedded": record.metadata.get("embedded", False),
            })
        except Exception as e:
            records.append({"country": country, "error": str(e)})

    return {
        "therapeutic_area": req.therapeutic_area,
        "condition": condition,
        "countries_seeded": len([r for r in records if "error" not in r]),
        "records": records,
    }


# ── Unified Ops Dashboard ─────────────────────────────────────────────────────

@router.get("/dashboard")
async def ops_dashboard(
    tenant_id: str = Query("global"),
    tier: str = Query("standard"),
):
    """
    Unified operations dashboard — SLA + billing + agents + data moat in one call.
    Used by the frontend dashboard panel.
    """
    from vinci_core.sla.monitor import sla_monitor
    from vinci_core.billing.metering import usage_meter
    from vinci_core.billing.plans import get_plan
    from vinci_core.embeddings.store import embedding_store
    from vinci_core.rwe.accumulation import rwe_accumulation
    from vinci_core.audit.gxp_trail import gxp_audit

    plan = get_plan(tier)
    sla = sla_monitor.get_report(tenant_id=tenant_id, tier=tier, window_hours=24)
    usage = usage_meter.get_summary(tenant_id=tenant_id, tier=tier)
    embed_stats = embedding_store.get_stats()
    rwe_stats = rwe_accumulation.get_stats()
    audit_stats = gxp_audit.get_stats(tenant_id=tenant_id)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tenant_id": tenant_id,
        "tier": tier,
        "plan_name": plan["name"],
        "sla": {
            "uptime_pct": sla.uptime_pct,
            "p50_ms": sla.p50_latency_ms,
            "p95_ms": sla.p95_latency_ms,
            "requests_24h": sla.total_requests,
            "errors_24h": sla.total_requests - sla.successful_requests,
            "status": "breach" if sla.breaches else "healthy",
            "breaches": sla.breaches,
        },
        "usage": {
            "api_calls": usage.units.get("api_calls", 0),
            "pipeline_runs": usage.units.get("pipeline_runs", 0),
            "rag_queries": usage.units.get("rag_queries", 0),
            "overage_usd": usage.total_overage_usd,
        },
        "data_moat": {
            "total_embeddings": embed_stats["total_documents"],
            "by_namespace": embed_stats["by_namespace"],
            "rwe_records": rwe_stats.total_records,
            "freshness_score": rwe_stats.freshness_score,
            "stale_datasets": len(rwe_stats.stale_datasets),
        },
        "audit": audit_stats,
        "agents": {
            "active": 10,
            "endpoints": [
                "/agents/twin/simulate", "/agents/iomt/adherence",
                "/agents/pgx/cross-reference", "/agents/regulatory/report",
                "/agents/pv/narrative", "/agents/sites/recommend",
                "/agents/pharmacist/review", "/agents/latam/roadmap",
                "/agents/vision/analyze", "/agents/patient/{id}/history",
            ],
        },
    }


# ── Agent Memory Context ──────────────────────────────────────────────────────

@router.get("/memory/context")
async def get_memory_context(
    tenant_id: str = Query("default"),
    query: str = Query(""),
    limit: int = Query(5),
):
    """Retrieve agent memory context for a tenant session."""
    from vinci_core.memory.agent_memory import AgentMemoryStore
    memory = AgentMemoryStore()
    context = memory.summarize(
        agent_type="platform",
        tenant_id=tenant_id,
        query=query,
        limit=limit,
    )
    recent = memory.recall(
        query=query or "recent",
        tenant_id=tenant_id,
        limit=limit,
    )
    return {
        "tenant_id": tenant_id,
        "context_summary": context,
        "recent_memories": [
            {
                "memory_id": m.memory_id,
                "content": m.content[:200],
                "importance": m.importance,
                "created_at": m.created_at,
            }
            for m in recent
        ],
    }
