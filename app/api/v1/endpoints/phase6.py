"""
Phase 6 API — Data Moat / Ariston AI.

LATAM health data access, semantic embedding search, RWE accumulation.

Endpoints:
  GET  /phase6/latam/sources              — available data sources per country
  GET  /phase6/latam/coverage             — population + record coverage stats
  GET  /phase6/latam/burden               — disease burden (PAHO/WHO estimates)
  POST /phase6/latam/fetch                — fetch epidemiological records
  POST /phase6/embeddings/upsert          — embed and store a document
  POST /phase6/embeddings/search          — semantic similarity search
  GET  /phase6/embeddings/stats           — embedding store statistics
  POST /phase6/rwe/accumulate             — trigger RWE accumulation job
  GET  /phase6/rwe/freshness              — data freshness status
  GET  /phase6/rwe/stats                  — accumulation pipeline stats
  POST /phase6/rwe/refresh                — re-fetch stale datasets
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger("ariston.phase6")
router = APIRouter(prefix="/phase6", tags=["Phase 6 — Data Moat"])


# ── Request models ───────────────────────────────────────────────────────────

class FetchEpiRequest(BaseModel):
    country: str
    condition: str
    year: Optional[int] = None
    region: Optional[str] = None


class DiseaseBurdenRequest(BaseModel):
    countries: list[str]
    conditions: list[str]


class EmbedUpsertRequest(BaseModel):
    content: str
    namespace: str = "rwe"
    doc_id: Optional[str] = None
    metadata: Optional[dict] = None


class EmbedSearchRequest(BaseModel):
    query: str
    namespace: str = "rwe"
    top_k: int = 5
    min_score: float = 0.1


class AccumulateRequest(BaseModel):
    country: str
    condition: str
    dataset_type: str = "epidemiological"
    year: Optional[int] = None
    namespace: str = "rwe"
    embed: bool = True


# ── LATAM data endpoints ─────────────────────────────────────────────────────

@router.get("/latam/sources")
async def get_data_sources(country: Optional[str] = Query(None)):
    """Available LATAM health data sources with dataset details."""
    from vinci_core.latam_data.connectors import latam_data_connector
    return {
        "sources": latam_data_connector.get_data_availability(country),
        "countries_supported": ["brazil", "mexico", "colombia", "argentina", "chile"],
    }


@router.get("/latam/coverage")
async def get_coverage():
    """Population coverage and record volume estimates across LATAM."""
    from vinci_core.latam_data.connectors import latam_data_connector
    stats = latam_data_connector.get_coverage_stats()
    total_pop = sum(v.get("population_millions", 0) or 0 for v in stats.values())
    total_records = sum(v.get("estimated_records_millions", 0) or 0 for v in stats.values())
    return {
        "countries": stats,
        "aggregate": {
            "total_population_millions": total_pop,
            "total_estimated_records_millions": total_records,
            "countries_count": len(stats),
        },
    }


@router.post("/latam/burden")
async def get_disease_burden(req: DiseaseBurdenRequest):
    """
    Disease burden estimates (prevalence + incidence per 100k) from PAHO/WHO data.
    Covers: type2_diabetes, cardiovascular, dengue, chagas_disease, oncology, tuberculosis, etc.
    """
    from vinci_core.latam_data.connectors import latam_data_connector
    if not req.countries:
        raise HTTPException(status_code=400, detail="At least one country required")
    if not req.conditions:
        raise HTTPException(status_code=400, detail="At least one condition required")

    burden = await latam_data_connector.fetch_disease_burden(
        countries=req.countries,
        conditions=req.conditions,
    )
    return {
        "burden": burden,
        "countries": req.countries,
        "conditions": req.conditions,
        "data_source": "PAHO/WHO 2022 estimates via DATASUS/SINAVE/SISPRO/SNVS/DEIS",
    }


@router.post("/latam/fetch")
async def fetch_epidemiological(req: FetchEpiRequest):
    """
    Fetch epidemiological records for a country + condition.
    Returns normalized DataRecords with ICD-10 codes, age groups, counts.
    """
    from vinci_core.latam_data.connectors import latam_data_connector
    from vinci_core.latam_data.connectors import LATAM_DATA_SOURCES

    if req.country.lower() not in LATAM_DATA_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported country: {req.country}. Supported: {list(LATAM_DATA_SOURCES.keys())}",
        )

    records = await latam_data_connector.fetch_epidemiological(
        country=req.country,
        condition=req.condition,
        year=req.year,
        region=req.region,
    )

    return {
        "country": req.country,
        "condition": req.condition,
        "year": req.year,
        "records": [
            {
                "record_id": r.record_id,
                "source": r.source,
                "dataset": r.dataset,
                "icd10_code": r.condition_code,
                "condition_name": r.condition_name,
                "count": r.count,
                "age_group": r.age_group,
                "sex": r.sex,
                "region": r.region,
                "year": r.year,
            }
            for r in records
        ],
        "total_records": len(records),
    }


# ── Embedding endpoints ──────────────────────────────────────────────────────

@router.post("/embeddings/upsert")
async def upsert_embedding(req: EmbedUpsertRequest):
    """
    Embed and store a document for semantic search.
    Content-addressed: identical content in same namespace returns existing doc.
    """
    from vinci_core.embeddings.store import embedding_store
    doc = embedding_store.upsert(
        content=req.content,
        namespace=req.namespace,
        doc_id=req.doc_id,
        metadata=req.metadata,
    )
    return {
        "doc_id": doc.doc_id,
        "namespace": doc.namespace,
        "content_hash": doc.content_hash,
        "embedding_dim": len(doc.embedding),
        "created_at": doc.created_at,
        "metadata": doc.metadata,
    }


@router.post("/embeddings/search")
async def search_embeddings(req: EmbedSearchRequest):
    """
    Semantic similarity search in an embedding namespace.
    Returns top-k most similar documents by cosine similarity.
    """
    from vinci_core.embeddings.store import embedding_store
    results = embedding_store.search(
        query=req.query,
        namespace=req.namespace,
        top_k=req.top_k,
        min_score=req.min_score,
    )
    return {
        "query": req.query,
        "namespace": req.namespace,
        "results": [
            {
                "doc_id": r.doc_id,
                "similarity_score": r.similarity_score,
                "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                "metadata": r.metadata,
            }
            for r in results
        ],
        "total_results": len(results),
    }


@router.get("/embeddings/stats")
async def get_embedding_stats(namespace: Optional[str] = Query(None)):
    """Embedding store statistics by namespace and provider."""
    from vinci_core.embeddings.store import embedding_store
    return embedding_store.get_stats(namespace=namespace)


# ── RWE accumulation endpoints ───────────────────────────────────────────────

@router.post("/rwe/accumulate")
async def accumulate_rwe(req: AccumulateRequest):
    """
    Trigger RWE accumulation: fetch LATAM data → embed → store.
    Powers the data flywheel for semantic similarity search.
    """
    from vinci_core.rwe.accumulation import rwe_accumulation
    record = await rwe_accumulation.accumulate(
        country=req.country,
        condition=req.condition,
        dataset_type=req.dataset_type,
        year=req.year,
        namespace=req.namespace,
        embed=req.embed,
    )
    return {
        "record_id": record.record_id,
        "country": record.country,
        "condition": record.condition,
        "dataset_type": record.dataset_type,
        "source": record.source,
        "doc_id": record.doc_id,
        "record_count": record.record_count,
        "year": record.year,
        "refreshed_at": record.refreshed_at,
        "embedded": record.metadata.get("embedded", False),
    }


@router.get("/rwe/freshness")
async def get_freshness(
    country: Optional[str] = Query(None),
    condition: Optional[str] = Query(None),
    dataset_type: Optional[str] = Query(None),
):
    """Data freshness status — shows stale vs fresh datasets."""
    from vinci_core.rwe.accumulation import rwe_accumulation
    statuses = rwe_accumulation.get_freshness(
        country=country,
        condition=condition,
        dataset_type=dataset_type,
    )
    stale = [s for s in statuses if s.is_stale]
    return {
        "total_datasets": len(statuses),
        "stale_count": len(stale),
        "fresh_count": len(statuses) - len(stale),
        "datasets": [
            {
                "dataset_type": s.dataset_type,
                "country": s.country,
                "condition": s.condition,
                "last_refreshed": s.last_refreshed,
                "age_hours": s.age_hours,
                "threshold_hours": s.threshold_hours,
                "is_stale": s.is_stale,
                "record_count": s.record_count,
            }
            for s in statuses
        ],
    }


@router.get("/rwe/stats")
async def get_rwe_stats():
    """RWE accumulation pipeline health metrics."""
    from vinci_core.rwe.accumulation import rwe_accumulation
    stats = rwe_accumulation.get_stats()
    return {
        "total_accumulation_records": stats.total_records,
        "namespaces": stats.total_namespaces,
        "by_country": stats.by_country,
        "by_dataset_type": stats.by_dataset_type,
        "stale_datasets": len(stats.stale_datasets),
        "freshness_score": stats.freshness_score,
        "data_flywheel_status": "healthy" if stats.freshness_score >= 0.8 else "stale",
    }


@router.post("/rwe/refresh")
async def refresh_stale_datasets(max_datasets: int = Query(10)):
    """
    Re-fetch all stale RWE datasets. Triggers accumulation pipeline.
    Useful for scheduled refresh jobs.
    """
    from vinci_core.rwe.accumulation import rwe_accumulation
    result = await rwe_accumulation.refresh_stale(max_datasets=max_datasets)
    return result
