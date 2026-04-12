"""
Phase 6 tests — Data Moat.

Tests: LATAM connectors, semantic embeddings, RWE accumulation pipeline,
       Phase 6 API endpoints.
"""

import os
import tempfile
import pytest
import asyncio


# ── LATAM data connectors ────────────────────────────────────────────────────

def test_latam_sources_all_countries():
    from vinci_core.latam_data.connectors import LATAM_DATA_SOURCES
    expected = {"brazil", "mexico", "colombia", "argentina", "chile"}
    assert set(LATAM_DATA_SOURCES.keys()) == expected


def test_latam_sources_have_required_fields():
    from vinci_core.latam_data.connectors import LATAM_DATA_SOURCES
    for country, meta in LATAM_DATA_SOURCES.items():
        assert "source" in meta, f"{country} missing source"
        assert "url" in meta, f"{country} missing url"
        assert "datasets" in meta, f"{country} missing datasets"
        assert "population_millions" in meta, f"{country} missing population_millions"


def test_get_data_availability_all():
    from vinci_core.latam_data.connectors import latam_data_connector
    avail = latam_data_connector.get_data_availability()
    assert len(avail) == 5


def test_get_data_availability_single():
    from vinci_core.latam_data.connectors import latam_data_connector
    avail = latam_data_connector.get_data_availability("brazil")
    assert "brazil" in avail
    assert avail["brazil"]["source"] == "DATASUS"


def test_get_data_availability_unknown():
    from vinci_core.latam_data.connectors import latam_data_connector
    avail = latam_data_connector.get_data_availability("atlantis")
    assert avail == {}


def test_get_coverage_stats():
    from vinci_core.latam_data.connectors import latam_data_connector
    stats = latam_data_connector.get_coverage_stats()
    assert len(stats) == 5
    for country, info in stats.items():
        assert "population_millions" in info
        assert "datasets" in info


@pytest.mark.asyncio
async def test_fetch_epidemiological_brazil():
    from vinci_core.latam_data.connectors import latam_data_connector
    records = await latam_data_connector.fetch_epidemiological(
        country="brazil",
        condition="type2_diabetes",
        year=2023,
    )
    assert len(records) > 0
    r = records[0]
    assert r.country == "brazil"
    assert r.condition_code == "E11"
    assert r.count > 0
    assert r.age_group is not None


@pytest.mark.asyncio
async def test_fetch_epidemiological_unknown_country():
    from vinci_core.latam_data.connectors import latam_data_connector
    records = await latam_data_connector.fetch_epidemiological(
        country="atlantis",
        condition="type2_diabetes",
    )
    assert records == []


@pytest.mark.asyncio
async def test_fetch_disease_burden():
    from vinci_core.latam_data.connectors import latam_data_connector
    burden = await latam_data_connector.fetch_disease_burden(
        countries=["brazil", "mexico"],
        conditions=["type2_diabetes", "cardiovascular"],
    )
    assert "brazil" in burden
    assert "mexico" in burden
    assert "type2_diabetes" in burden["brazil"]
    assert burden["brazil"]["type2_diabetes"]["prevalence_per_100k"] is not None


@pytest.mark.asyncio
async def test_fetch_disease_burden_missing_condition():
    from vinci_core.latam_data.connectors import latam_data_connector
    burden = await latam_data_connector.fetch_disease_burden(
        countries=["chile"],
        conditions=["rare_condition_xyz"],
    )
    assert burden["chile"]["rare_condition_xyz"]["prevalence_per_100k"] is None
    assert burden["chile"]["rare_condition_xyz"]["source"] == "Not available"


def test_datarecord_fields():
    from vinci_core.latam_data.connectors import DataRecord
    r = DataRecord(
        record_id="test",
        country="brazil",
        source="DATASUS",
        dataset="AIH",
        year=2023,
        condition_code="E11",
        condition_name="Type 2 diabetes",
        count=500,
        region=None,
        age_group="45-59",
        sex="all",
    )
    assert r.retrieved_at is not None


# ── Semantic embeddings ──────────────────────────────────────────────────────

@pytest.fixture
def temp_embed_db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "test_embed.db")


def test_embed_upsert(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    doc = store.upsert("Diabetes prevalence in Brazil 2023", namespace="rwe")
    assert doc.doc_id is not None
    assert len(doc.embedding) in (256, 384)  # TF-IDF 256-dim or sentence-transformers 384-dim
    assert doc.namespace == "rwe"


def test_embed_deduplication(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    doc1 = store.upsert("Same content here", namespace="test")
    doc2 = store.upsert("Same content here", namespace="test")
    assert doc1.doc_id == doc2.doc_id  # content-addressed dedup


def test_embed_different_namespaces(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    doc1 = store.upsert("Same content here", namespace="ns_a")
    doc2 = store.upsert("Same content here", namespace="ns_b")
    # Different namespaces → different docs
    assert doc1.doc_id != doc2.doc_id


def test_embed_search_returns_results(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    store.upsert("Type 2 diabetes mellitus management in LATAM patients", namespace="rwe")
    store.upsert("Cardiovascular disease risk factors in Mexico", namespace="rwe")
    results = store.search("diabetes treatment", namespace="rwe", top_k=5)
    assert len(results) >= 1
    assert results[0].similarity_score > 0


def test_embed_search_empty_namespace(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    results = store.search("diabetes", namespace="empty_ns")
    assert results == []


def test_embed_search_respects_top_k(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    for i in range(10):
        store.upsert(f"Medical document {i} about diabetes treatment protocols", namespace="test")
    results = store.search("diabetes", namespace="test", top_k=3)
    assert len(results) <= 3


def test_embed_delete(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    doc = store.upsert("Content to delete", namespace="del_test")
    deleted = store.delete(doc.doc_id)
    assert deleted is True
    # Should not appear in search
    results = store.search("Content to delete", namespace="del_test")
    assert not any(r.doc_id == doc.doc_id for r in results)


def test_embed_get_stats(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    store.upsert("Doc 1 about clinical trials", namespace="rwe")
    store.upsert("Doc 2 about pharmacovigilance", namespace="regulatory")
    stats = store.get_stats()
    assert stats["total_documents"] == 2
    assert "rwe" in stats["by_namespace"]
    assert "regulatory" in stats["by_namespace"]
    assert "active_provider" in stats


def test_tfidf_embedding_deterministic(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    vec1 = store._embed_tfidf("diabetes treatment protocol LATAM")
    vec2 = store._embed_tfidf("diabetes treatment protocol LATAM")
    assert vec1 == vec2  # deterministic


def test_tfidf_embedding_normalized(temp_embed_db):
    import math
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    vec = store._embed_tfidf("some medical text about patient outcomes")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 0.01  # L2 normalized


def test_cosine_similarity_identical():
    from vinci_core.embeddings.store import EmbeddingStore
    vec = [0.5, 0.5, 0.5, 0.5]
    score = EmbeddingStore._cosine_similarity(vec, vec)
    assert abs(score - 1.0) < 0.001


def test_cosine_similarity_orthogonal():
    from vinci_core.embeddings.store import EmbeddingStore
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    score = EmbeddingStore._cosine_similarity(a, b)
    assert abs(score) < 0.001


def test_cosine_similarity_dim_mismatch():
    from vinci_core.embeddings.store import EmbeddingStore
    score = EmbeddingStore._cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
    assert score == 0.0


def test_embed_with_metadata(temp_embed_db):
    from vinci_core.embeddings.store import EmbeddingStore
    store = EmbeddingStore(db_path=temp_embed_db)
    doc = store.upsert(
        "Clinical evidence for GLP-1 agonists in T2D",
        namespace="drug_discovery",
        metadata={"country": "brazil", "condition": "type2_diabetes"},
    )
    assert doc.metadata["country"] == "brazil"
    results = store.search("GLP-1 diabetes", namespace="drug_discovery", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["country"] == "brazil"


# ── RWE accumulation ─────────────────────────────────────────────────────────

@pytest.fixture
def temp_rwe_db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "test_rwe.db")


@pytest.mark.asyncio
async def test_accumulate_basic(temp_rwe_db, tmp_path):
    from vinci_core.rwe.accumulation import RWEAccumulationPipeline
    pipeline = RWEAccumulationPipeline(db_path=temp_rwe_db)
    record = await pipeline.accumulate(
        country="brazil",
        condition="type2_diabetes",
        dataset_type="epidemiological",
        embed=False,  # skip embedding for speed
    )
    assert record.country == "brazil"
    assert record.condition == "type2_diabetes"
    assert record.record_count > 0
    assert record.doc_id is not None


@pytest.mark.asyncio
async def test_accumulate_multiple(temp_rwe_db):
    from vinci_core.rwe.accumulation import RWEAccumulationPipeline
    pipeline = RWEAccumulationPipeline(db_path=temp_rwe_db)
    for country in ["brazil", "mexico"]:
        await pipeline.accumulate(country=country, condition="dengue", embed=False)
    stats = pipeline.get_stats()
    assert stats.total_records >= 2


def test_freshness_empty(temp_rwe_db):
    from vinci_core.rwe.accumulation import RWEAccumulationPipeline
    pipeline = RWEAccumulationPipeline(db_path=temp_rwe_db)
    statuses = pipeline.get_freshness()
    assert statuses == []


@pytest.mark.asyncio
async def test_freshness_after_accumulate(temp_rwe_db):
    from vinci_core.rwe.accumulation import RWEAccumulationPipeline
    pipeline = RWEAccumulationPipeline(db_path=temp_rwe_db)
    await pipeline.accumulate(country="chile", condition="cardiovascular", embed=False)
    statuses = pipeline.get_freshness()
    assert len(statuses) > 0
    # Just accumulated — should be fresh
    fresh = [s for s in statuses if not s.is_stale]
    assert len(fresh) > 0


def test_get_stats_empty(temp_rwe_db):
    from vinci_core.rwe.accumulation import RWEAccumulationPipeline
    pipeline = RWEAccumulationPipeline(db_path=temp_rwe_db)
    stats = pipeline.get_stats()
    assert stats.total_records == 0
    assert stats.freshness_score == 1.0


def test_purge_old(temp_rwe_db):
    from vinci_core.rwe.accumulation import RWEAccumulationPipeline
    pipeline = RWEAccumulationPipeline(db_path=temp_rwe_db)
    deleted = pipeline.purge_old(retention_days=365)
    assert deleted >= 0


def test_freshness_thresholds():
    from vinci_core.rwe.accumulation import FRESHNESS_THRESHOLDS
    assert "epidemiological" in FRESHNESS_THRESHOLDS
    assert "pubmed" in FRESHNESS_THRESHOLDS
    assert FRESHNESS_THRESHOLDS["pubmed"] < FRESHNESS_THRESHOLDS["disease_burden"]


# ── Phase 6 API endpoints ────────────────────────────────────────────────────

@pytest.fixture
def client_p6():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.v1.endpoints.phase6 import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=True)


def test_get_latam_sources(client_p6):
    r = client_p6.get("/phase6/latam/sources")
    assert r.status_code == 200
    data = r.json()
    assert "sources" in data
    assert "brazil" in data["sources"]
    assert "countries_supported" in data


def test_get_latam_sources_single_country(client_p6):
    r = client_p6.get("/phase6/latam/sources?country=mexico")
    assert r.status_code == 200
    assert "mexico" in r.json()["sources"]


def test_get_latam_coverage(client_p6):
    r = client_p6.get("/phase6/latam/coverage")
    assert r.status_code == 200
    data = r.json()
    assert "aggregate" in data
    assert data["aggregate"]["total_population_millions"] > 400


def test_get_disease_burden(client_p6):
    r = client_p6.post("/phase6/latam/burden", json={
        "countries": ["brazil", "mexico"],
        "conditions": ["type2_diabetes", "cardiovascular"],
    })
    assert r.status_code == 200
    data = r.json()
    assert "burden" in data
    assert "brazil" in data["burden"]
    assert "type2_diabetes" in data["burden"]["brazil"]


def test_fetch_epidemiological(client_p6):
    r = client_p6.post("/phase6/latam/fetch", json={
        "country": "brazil",
        "condition": "type2_diabetes",
        "year": 2023,
    })
    assert r.status_code == 200
    data = r.json()
    assert "records" in data
    assert data["total_records"] > 0
    assert data["records"][0]["icd10_code"] == "E11"


def test_fetch_epidemiological_unknown_country(client_p6):
    r = client_p6.post("/phase6/latam/fetch", json={
        "country": "atlantis",
        "condition": "type2_diabetes",
    })
    assert r.status_code == 400


def test_embed_upsert_endpoint(client_p6):
    r = client_p6.post("/phase6/embeddings/upsert", json={
        "content": "Clinical evidence for SGLT2 inhibitors in diabetic nephropathy LATAM",
        "namespace": "rwe",
    })
    assert r.status_code == 200
    data = r.json()
    assert "doc_id" in data
    assert "embedding_dim" in data
    assert data["embedding_dim"] in (256, 384)  # TF-IDF or sentence-transformers


def test_embed_search_endpoint(client_p6):
    # First upsert
    client_p6.post("/phase6/embeddings/upsert", json={
        "content": "Chagas disease treatment trypanosoma cruzi LATAM",
        "namespace": "drug_discovery",
    })
    # Then search
    r = client_p6.post("/phase6/embeddings/search", json={
        "query": "Chagas trypanosoma treatment",
        "namespace": "drug_discovery",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "total_results" in data


def test_embed_stats_endpoint(client_p6):
    r = client_p6.get("/phase6/embeddings/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_documents" in data
    assert "active_provider" in data


def test_rwe_accumulate_endpoint(client_p6):
    r = client_p6.post("/phase6/rwe/accumulate", json={
        "country": "colombia",
        "condition": "dengue",
        "embed": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["country"] == "colombia"
    assert data["condition"] == "dengue"
    assert data["record_count"] > 0


def test_rwe_freshness_endpoint(client_p6):
    r = client_p6.get("/phase6/rwe/freshness")
    assert r.status_code == 200
    data = r.json()
    assert "total_datasets" in data
    assert "stale_count" in data


def test_rwe_stats_endpoint(client_p6):
    r = client_p6.get("/phase6/rwe/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_accumulation_records" in data
    assert "freshness_score" in data
    assert "data_flywheel_status" in data


def test_rwe_refresh_endpoint(client_p6):
    r = client_p6.post("/phase6/rwe/refresh?max_datasets=5")
    assert r.status_code == 200
    data = r.json()
    assert "stale_found" in data
    assert "refreshed" in data
    assert "timestamp" in data
