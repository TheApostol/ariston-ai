"""
Tests for new components:
  - LatAm Localization Layer
  - Continuous Improvement Loop
  - LatAm Pilot Program Framework
"""

import json
import os
import pytest


# ── Localization: Regulatory Mapping ─────────────────────────────────────────

from app.localization.regulatory_mapping import (
    map_requirement,
    get_all_mappings_for_locale,
    get_agency_for_locale,
    get_locale_config,
    REGULATORY_CROSSWALK,
)


def test_map_nda_to_anvisa():
    result = map_requirement("NDA", "ANVISA")
    assert result is not None
    assert "equivalent" in result
    assert "timeline_days" in result
    assert result["timeline_days"] > 0


def test_map_ind_to_cofepris():
    result = map_requirement("IND", "COFEPRIS")
    assert result is not None
    assert "equivalent" in result
    assert "COFEPRIS" in result.get("pathway", "") or "Autorización" in result["equivalent"]


def test_map_unknown_returns_none():
    result = map_requirement("UNKNOWN_REQ", "ANVISA")
    assert result is None


def test_map_known_req_unknown_agency():
    result = map_requirement("NDA", "UNKNOWN_AGENCY")
    assert result is None


def test_get_agency_for_locale():
    assert get_agency_for_locale("pt-BR") == "ANVISA"
    assert get_agency_for_locale("es-MX") == "COFEPRIS"
    assert get_agency_for_locale("es-CO") == "INVIMA"
    assert get_agency_for_locale("es-AR") == "ANMAT"
    assert get_agency_for_locale("en-US") == "FDA"


def test_get_locale_config_brazil():
    config = get_locale_config("pt-BR")
    assert config["currency"] == "BRL"
    assert config["currency_symbol"] == "R$"
    assert config["date_format"] == "DD/MM/YYYY"


def test_get_locale_config_mexico():
    config = get_locale_config("es-MX")
    assert config["currency"] == "MXN"


def test_get_all_mappings_for_locale():
    mappings = get_all_mappings_for_locale("pt-BR")
    assert len(mappings) > 0
    for mapping in mappings.values():
        assert "equivalent" in mapping


def test_all_crosswalk_entries_have_required_fields():
    required_fields = ["equivalent", "reference", "pathway", "timeline_days", "notes"]
    for fda_req, agency_map in REGULATORY_CROSSWALK.items():
        for agency, mapping in agency_map.items():
            for field in required_fields:
                assert field in mapping, (
                    f"Missing '{field}' in {fda_req} → {agency} mapping"
                )


# ── Localization: Language Detection ─────────────────────────────────────────

from app.localization.service import detect_language


def test_detect_english():
    text = "The patient experienced serious adverse events during the clinical trial."
    assert detect_language(text) == "en"


def test_detect_portuguese():
    text = "O paciente não apresentou reações adversas no ensaio clínico."
    assert detect_language(text) == "pt-BR"


def test_detect_spanish():
    text = "El paciente no presentó reacciones adversas durante el ensayo clínico con el medicamento."
    assert detect_language(text) == "es"


# ── Localization: Partner Database ───────────────────────────────────────────

from app.localization.partner_db import (
    get_partners,
    get_partner_by_id,
    list_countries,
    list_specialties,
    PARTNER_DATABASE,
)


def test_all_partners_have_required_fields():
    required = ["id", "name", "country", "locale", "agency", "type", "specialties", "city"]
    for partner in PARTNER_DATABASE:
        for field in required:
            assert field in partner, f"Partner {partner.get('id')} missing '{field}'"


def test_get_partners_by_country():
    brazil = get_partners(country="Brazil")
    assert len(brazil) > 0
    for p in brazil:
        assert "Brazil" in p["country"]


def test_get_partners_by_agency():
    anvisa = get_partners(agency="ANVISA")
    assert len(anvisa) > 0


def test_get_partners_by_specialty():
    oncology = get_partners(specialty="oncology")
    assert len(oncology) > 0
    for p in oncology:
        assert any("oncology" in s for s in p["specialties"])


def test_get_partner_by_id_found():
    partner = get_partner_by_id("br-001")
    assert partner is not None
    assert partner["country"] == "Brazil"


def test_get_partner_by_id_not_found():
    partner = get_partner_by_id("nonexistent-999")
    assert partner is None


def test_list_countries():
    countries = list_countries()
    assert "Brazil" in countries
    assert "Mexico" in countries
    assert "Colombia" in countries
    assert "Argentina" in countries


def test_list_specialties():
    specialties = list_specialties()
    assert "oncology" in specialties
    assert len(specialties) > 5


# ── Continuous Improvement: Benchmark Analyzer ───────────────────────────────

from vinci_core.continuous_improvement.benchmark_analyzer import (
    analyze_benchmarks,
    get_low_scoring_patterns,
    _combined_score,
)


def test_combined_score_calculation():
    metrics = {"safety_score": 1.0, "grounding_score": 0.75, "confidence_score": 0.90}
    expected = 1.0 * 0.50 + 0.75 * 0.30 + 0.90 * 0.20
    assert abs(_combined_score(metrics) - expected) < 0.001


def test_combined_score_all_zeros():
    assert _combined_score({}) == 0.0


def test_analyze_benchmarks_empty():
    result = analyze_benchmarks("/tmp/nonexistent_eval.jsonl")
    assert result["summary"]["total_entries"] == 0
    assert result["flags"] == []
    assert result["by_layer"] == {}


def test_analyze_benchmarks_with_data(tmp_path):
    log_file = str(tmp_path / "eval.jsonl")
    entries = [
        {
            "layer": "pharma", "model": "test-model", "safety_flag": "SAFE",
            "metrics": {"safety_score": 1.0, "grounding_score": 0.8, "confidence_score": 0.9},
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "layer": "pharma", "model": "test-model", "safety_flag": "SAFE",
            "metrics": {"safety_score": 1.0, "grounding_score": 0.7, "confidence_score": 0.85},
            "timestamp": "2024-01-01T00:01:00Z",
        },
        {
            "layer": "clinical", "model": "weak-model", "safety_flag": "FLAGGED",
            "metrics": {"safety_score": 0.0, "grounding_score": 0.3, "confidence_score": 0.4},
            "timestamp": "2024-01-01T00:02:00Z",
        },
    ]
    with open(log_file, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    result = analyze_benchmarks(log_file, threshold=0.75)
    assert result["summary"]["total_entries"] == 3
    assert "pharma" in result["by_layer"]
    assert "clinical" in result["by_layer"]
    assert result["summary"]["safety_failure_count"] == 1


# ── Continuous Improvement: Feedback Loop ────────────────────────────────────

from vinci_core.continuous_improvement.feedback_loop import (
    submit_feedback,
    get_feedback_summary,
    get_unprocessed_signals,
    mark_signal_processed,
)


@pytest.fixture
def feedback_db(tmp_path):
    return str(tmp_path / "feedback.db")


def test_submit_good_feedback_no_signal(feedback_db):
    result = submit_feedback(rating=5, layer="pharma", comment="Excellent!", db_path=feedback_db)
    assert result["status"] == "recorded"
    assert result["signal_generated"] is False


def test_submit_bad_feedback_generates_signal(feedback_db):
    result = submit_feedback(rating=2, layer="clinical", comment="Too vague", db_path=feedback_db)
    assert result["signal_generated"] is True
    signals = get_unprocessed_signals(db_path=feedback_db)
    assert len(signals) == 1


def test_mark_signal_processed(feedback_db):
    submit_feedback(rating=1, layer="pharma", db_path=feedback_db)
    signals = get_unprocessed_signals(db_path=feedback_db)
    assert len(signals) == 1
    mark_signal_processed(signals[0]["id"], db_path=feedback_db)
    signals_after = get_unprocessed_signals(db_path=feedback_db)
    assert len(signals_after) == 0


def test_feedback_summary_empty(feedback_db):
    summary = get_feedback_summary(db_path=feedback_db)
    assert summary["total_responses"] == 0
    assert summary["avg_rating"] is None


def test_feedback_summary_with_data(feedback_db):
    submit_feedback(rating=5, layer="pharma", db_path=feedback_db)
    submit_feedback(rating=3, layer="clinical", db_path=feedback_db)
    summary = get_feedback_summary(db_path=feedback_db)
    assert summary["total_responses"] == 2
    assert summary["avg_rating"] == 4.0


# ── Pilot Programs ────────────────────────────────────────────────────────────

from app.pilot_programs.service import (
    enroll_pilot,
    get_pilot,
    list_pilots,
    update_pilot_status,
    save_document_version,
    get_document_versions,
    get_document_content,
    record_roi_metric,
    get_roi_summary,
    submit_pilot_feedback,
    get_pilot_feedback,
    get_all_pilots_analytics,
)


@pytest.fixture
def pilot_db(tmp_path):
    return str(tmp_path / "pilots.db")


@pytest.fixture
def enrolled_pilot(pilot_db):
    result = enroll_pilot(
        company_name="BioPharma MX",
        contact_name="Dr. González",
        contact_email="dr@biopharma.mx",
        country="Mexico",
        locale="es-MX",
        agency="COFEPRIS",
        therapeutic_area="oncology",
        commitment_level="trial",
        db_path=pilot_db,
    )
    return result["pilot_id"]


def test_enroll_pilot(pilot_db):
    result = enroll_pilot(
        company_name="TestPharma",
        contact_name="Test User",
        contact_email="test@example.com",
        country="Brazil",
        locale="pt-BR",
        agency="ANVISA",
        therapeutic_area="rare_disease",
        commitment_level="committed",
        db_path=pilot_db,
    )
    assert result["status"] == "enrolled"
    assert "pilot_id" in result
    assert len(result["pilot_id"]) > 0


def test_get_pilot(pilot_db, enrolled_pilot):
    pilot = get_pilot(enrolled_pilot, db_path=pilot_db)
    assert pilot is not None
    assert pilot["company_name"] == "BioPharma MX"
    assert pilot["agency"] == "COFEPRIS"


def test_get_pilot_not_found(pilot_db):
    pilot = get_pilot("nonexistent-id", db_path=pilot_db)
    assert pilot is None


def test_list_pilots(pilot_db, enrolled_pilot):
    pilots = list_pilots(db_path=pilot_db)
    assert len(pilots) >= 1


def test_update_pilot_status(pilot_db, enrolled_pilot):
    update_pilot_status(enrolled_pilot, "paused", db_path=pilot_db)
    pilot = get_pilot(enrolled_pilot, db_path=pilot_db)
    assert pilot["status"] == "paused"


def test_document_versioning(pilot_db, enrolled_pilot):
    v1 = save_document_version(
        pilot_id=enrolled_pilot,
        document_type="csr",
        content="Version 1 content",
        db_path=pilot_db,
    )
    assert v1["version"] == 1

    v2 = save_document_version(
        pilot_id=enrolled_pilot,
        document_type="csr",
        content="Version 2 content",
        change_summary="Updated safety section",
        db_path=pilot_db,
    )
    assert v2["version"] == 2

    versions = get_document_versions(enrolled_pilot, db_path=pilot_db)
    assert len(versions) == 2


def test_document_content_retrieval(pilot_db, enrolled_pilot):
    v = save_document_version(
        pilot_id=enrolled_pilot,
        document_type="ectd",
        content="eCTD Module 5 content",
        db_path=pilot_db,
    )
    doc = get_document_content(v["document_id"], db_path=pilot_db)
    assert doc is not None
    assert "eCTD" in doc["content"]


def test_roi_calculation(pilot_db, enrolled_pilot):
    result = record_roi_metric(
        pilot_id=enrolled_pilot,
        document_type="csr",
        manual_hours_baseline=40.0,
        ai_assisted_hours=8.0,
        hourly_rate_usd=150.0,
        db_path=pilot_db,
    )
    assert result["time_saved_hours"] == 32.0
    assert result["cost_saved_usd"] == 4800.0
    assert result["time_reduction_pct"] == 80.0


def test_roi_summary(pilot_db, enrolled_pilot):
    record_roi_metric(
        pilot_id=enrolled_pilot,
        document_type="csr",
        manual_hours_baseline=40.0,
        ai_assisted_hours=8.0,
        db_path=pilot_db,
    )
    summary = get_roi_summary(enrolled_pilot, db_path=pilot_db)
    assert summary["total_cost_saved_usd"] > 0
    assert summary["avg_time_reduction_pct"] > 0


def test_roi_summary_empty(pilot_db, enrolled_pilot):
    summary = get_roi_summary(enrolled_pilot, db_path=pilot_db)
    assert summary["total_documents"] == 0
    assert summary["total_cost_saved_usd"] == 0


def test_pilot_feedback(pilot_db, enrolled_pilot):
    result = submit_pilot_feedback(
        pilot_id=enrolled_pilot,
        rating=4,
        nps_score=8,
        feature_ratings={"csr_drafting": 5},
        comment="Works well",
        db_path=pilot_db,
    )
    assert result["status"] == "recorded"

    feedbacks = get_pilot_feedback(enrolled_pilot, db_path=pilot_db)
    assert len(feedbacks) == 1
    assert feedbacks[0]["feature_ratings"]["csr_drafting"] == 5


def test_platform_analytics(pilot_db, enrolled_pilot):
    analytics = get_all_pilots_analytics(db_path=pilot_db)
    assert analytics["total_pilots"] >= 1
    assert "Mexico" in analytics["countries"]
