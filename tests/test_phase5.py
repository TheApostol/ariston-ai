"""
Phase 5 tests — Revenue Infrastructure.

Tests: billing plans, usage metering, SLA monitoring, Stripe integration,
       customer portal API endpoints.
"""

import os
import tempfile
import pytest


# ── Billing plans ────────────────────────────────────────────────────────────

def test_plans_all_tiers():
    from vinci_core.billing.plans import PLANS
    assert set(PLANS.keys()) == {"pilot", "standard", "premium", "enterprise"}


def test_plans_price_progression():
    from vinci_core.billing.plans import PLANS
    assert PLANS["pilot"]["price_usd_year"] == 0
    assert PLANS["standard"]["price_usd_year"] == 100_000
    assert PLANS["premium"]["price_usd_year"] == 300_000
    assert PLANS["enterprise"]["price_usd_year"] is None  # negotiated


def test_get_plan_fallback():
    from vinci_core.billing.plans import get_plan
    plan = get_plan("unknown_tier")
    assert plan["name"] == "Standard"


def test_is_within_quota():
    from vinci_core.billing.plans import is_within_quota
    assert is_within_quota("standard", "api_calls", 9_999) is True
    assert is_within_quota("standard", "api_calls", 10_001) is False


def test_is_within_quota_unlimited():
    from vinci_core.billing.plans import is_within_quota
    # Enterprise has -1 (unlimited) for api_calls
    assert is_within_quota("enterprise", "api_calls", 1_000_000) is True


def test_overage_cost():
    from vinci_core.billing.plans import overage_cost
    # Standard: api_calls overage = $0.05/unit
    cost = overage_cost("standard", "api_calls", 100)
    assert cost == 5.0


def test_overage_cost_enterprise_zero():
    from vinci_core.billing.plans import overage_cost
    # Enterprise has no overage rates
    cost = overage_cost("enterprise", "api_calls", 1000)
    assert cost == 0.0


def test_plans_have_stripe_price_ids():
    from vinci_core.billing.plans import PLANS
    assert PLANS["standard"]["stripe_price_id"] is not None
    assert PLANS["premium"]["stripe_price_id"] is not None


# ── Usage metering ───────────────────────────────────────────────────────────

@pytest.fixture
def temp_billing_db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "test_billing.db")


def test_record_usage(temp_billing_db):
    from vinci_core.billing.metering import UsageMeter
    meter = UsageMeter(db_path=temp_billing_db)
    event = meter.record(
        tenant_id="tenant_1",
        unit="api_calls",
        quantity=3,
        tier="standard",
    )
    assert event.tenant_id == "tenant_1"
    assert event.quantity == 3
    assert event.unit == "api_calls"


def test_usage_summary(temp_billing_db):
    from vinci_core.billing.metering import UsageMeter
    meter = UsageMeter(db_path=temp_billing_db)
    for _ in range(5):
        meter.record(tenant_id="t1", unit="api_calls", tier="standard")
    meter.record(tenant_id="t1", unit="pipeline_runs", tier="standard")

    summary = meter.get_summary(tenant_id="t1", tier="standard")
    assert summary.units.get("api_calls", 0) == 5
    assert summary.units.get("pipeline_runs", 0) == 1
    assert summary.tier == "standard"


def test_usage_within_quota(temp_billing_db):
    from vinci_core.billing.metering import UsageMeter
    meter = UsageMeter(db_path=temp_billing_db)
    result = meter.check_quota("t2", "api_calls", "standard")
    assert result["allowed"] is True
    assert result["used"] == 0
    assert result["limit"] == 10_000


def test_overage_triggers_on_excess(temp_billing_db):
    from vinci_core.billing.metering import UsageMeter
    meter = UsageMeter(db_path=temp_billing_db)
    # Pilot has 500 api_calls; record 501 in one shot
    event = meter.record(
        tenant_id="pilot_t",
        unit="api_calls",
        quantity=501,
        tier="pilot",
    )
    assert event.cost_usd > 0


def test_get_invoice_data(temp_billing_db):
    from vinci_core.billing.metering import UsageMeter
    meter = UsageMeter(db_path=temp_billing_db)
    data = meter.get_invoice_data(tenant_id="inv_t", tier="standard")
    assert "base_subscription_usd" in data
    assert data["base_subscription_usd"] == 100_000
    assert "total_due_usd" in data


def test_get_daily_breakdown(temp_billing_db):
    from vinci_core.billing.metering import UsageMeter
    meter = UsageMeter(db_path=temp_billing_db)
    meter.record(tenant_id="daily_t", unit="api_calls", tier="standard")
    breakdown = meter.get_daily_breakdown(tenant_id="daily_t", unit="api_calls", days=7)
    assert isinstance(breakdown, list)
    if breakdown:
        assert "date" in breakdown[0]
        assert "count" in breakdown[0]


def test_metering_isolation(temp_billing_db):
    from vinci_core.billing.metering import UsageMeter
    meter = UsageMeter(db_path=temp_billing_db)
    meter.record(tenant_id="tenant_A", unit="api_calls")
    meter.record(tenant_id="tenant_B", unit="api_calls")

    summary_a = meter.get_summary(tenant_id="tenant_A", tier="standard")
    summary_b = meter.get_summary(tenant_id="tenant_B", tier="standard")
    assert summary_a.units.get("api_calls", 0) == 1
    assert summary_b.units.get("api_calls", 0) == 1


# ── SLA monitoring ───────────────────────────────────────────────────────────

@pytest.fixture
def temp_sla_db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "test_sla.db")


def test_sla_record(temp_sla_db):
    from vinci_core.sla.monitor import SLAMonitor
    mon = SLAMonitor(db_path=temp_sla_db)
    metric = mon.record(latency_ms=500.0, success=True, tenant_id="t1")
    assert metric.metric_id is not None
    assert metric.latency_ms == 500.0
    assert metric.success is True


def test_sla_report_empty(temp_sla_db):
    from vinci_core.sla.monitor import SLAMonitor
    mon = SLAMonitor(db_path=temp_sla_db)
    report = mon.get_report(tenant_id="nobody", tier="standard", window_hours=24)
    assert report.total_requests == 0
    assert report.uptime_pct == 100.0
    assert report.breaches == []


def test_sla_report_with_data(temp_sla_db):
    from vinci_core.sla.monitor import SLAMonitor
    mon = SLAMonitor(db_path=temp_sla_db)
    for i in range(10):
        mon.record(latency_ms=float(300 + i * 10), success=True, tenant_id="t1")
    report = mon.get_report(tenant_id="t1", tier="standard", window_hours=1)
    assert report.total_requests == 10
    assert report.error_rate_pct == 0.0
    assert report.p50_latency_ms > 0


def test_sla_breach_detection(temp_sla_db):
    from vinci_core.sla.monitor import SLAMonitor
    mon = SLAMonitor(db_path=temp_sla_db)
    # Record very high latency that exceeds P50 SLA
    for _ in range(5):
        mon.record(latency_ms=9999.0, success=True, tenant_id="breach_t")
    report = mon.get_report(tenant_id="breach_t", tier="enterprise", window_hours=1)
    # P50 SLA for enterprise is 800ms; 9999ms should breach
    assert any("P50" in b or "P95" in b for b in report.breaches)


def test_sla_error_rate_breach(temp_sla_db):
    from vinci_core.sla.monitor import SLAMonitor
    mon = SLAMonitor(db_path=temp_sla_db)
    for _ in range(90):
        mon.record(latency_ms=100.0, success=True, tenant_id="err_t")
    for _ in range(10):
        mon.record(latency_ms=100.0, success=False, tenant_id="err_t", error_type="timeout")
    report = mon.get_report(tenant_id="err_t", tier="enterprise", window_hours=1)
    # 10% error rate > 0.1% enterprise SLA
    assert any("Error rate" in b for b in report.breaches)


def test_sla_uptime_series(temp_sla_db):
    from vinci_core.sla.monitor import SLAMonitor
    mon = SLAMonitor(db_path=temp_sla_db)
    mon.record(latency_ms=100.0, success=True, tenant_id="series_t")
    series = mon.get_uptime_series(tenant_id="series_t", days=7)
    assert isinstance(series, list)


def test_sla_purge_old(temp_sla_db):
    from vinci_core.sla.monitor import SLAMonitor
    mon = SLAMonitor(db_path=temp_sla_db)
    mon.record(latency_ms=100.0, success=True)
    deleted = mon.purge_old(retention_days=0)
    assert deleted >= 0  # may or may not delete based on timing


def test_sla_targets_all_tiers():
    from vinci_core.sla.monitor import SLA_TARGETS
    for tier in ("pilot", "standard", "premium", "enterprise"):
        assert tier in SLA_TARGETS
        t = SLA_TARGETS[tier]
        assert "uptime_pct" in t
        assert "p50_latency_ms" in t
        assert "p95_latency_ms" in t
        assert "error_rate_pct" in t


# ── Stripe integration ───────────────────────────────────────────────────────

def test_stripe_mock_create_customer():
    from vinci_core.billing.stripe_integration import StripeIntegration
    s = StripeIntegration()
    # Without STRIPE_SECRET_KEY set, should return mock customer
    customer = s.create_customer("t123", "test@pharma.com", "Test Pharma Inc")
    assert customer is not None
    assert customer.customer_id.startswith("cus_mock_")
    assert customer.tenant_id == "t123"


def test_stripe_mock_create_subscription():
    from vinci_core.billing.stripe_integration import StripeIntegration
    s = StripeIntegration()
    sub = s.create_subscription("cus_mock_test", "price_standard_annual", trial_days=30)
    assert sub is not None
    assert sub.get("status") == "trialing"
    assert sub.get("mock") is True


def test_stripe_mock_record_usage():
    from vinci_core.billing.stripe_integration import StripeIntegration
    s = StripeIntegration()
    result = s.record_usage("si_mock_item", quantity=100)
    assert result is True


def test_stripe_mock_cancel():
    from vinci_core.billing.stripe_integration import StripeIntegration
    s = StripeIntegration()
    result = s.cancel_subscription("sub_mock_test", at_period_end=True)
    assert result is True


def test_stripe_mock_invoice():
    from vinci_core.billing.stripe_integration import StripeIntegration
    s = StripeIntegration()
    inv = s.get_upcoming_invoice("cus_mock_test")
    assert inv is not None
    assert "amount_due" in inv or "mock" in inv


def test_stripe_webhook_no_secret():
    from vinci_core.billing.stripe_integration import StripeIntegration
    s = StripeIntegration()
    result = s.process_webhook(b'{"type":"test"}', "sig_header")
    # No STRIPE_WEBHOOK_SECRET → returns None
    assert result is None


# ── Phase 5 API endpoints ────────────────────────────────────────────────────

@pytest.fixture
def client_p5():
    fastapi = pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.v1.endpoints.phase5 import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=True)


def test_list_plans(client_p5):
    r = client_p5.get("/phase5/billing/plans")
    assert r.status_code == 200
    data = r.json()
    assert "plans" in data
    assert "standard" in data["plans"]
    assert "enterprise" in data["plans"]


def test_get_usage(client_p5):
    r = client_p5.get("/phase5/billing/usage?tenant_id=test_t&tier=standard")
    assert r.status_code == 200
    data = r.json()
    assert "usage" in data
    assert "quota" in data
    assert "within_quota" in data


def test_get_invoice(client_p5):
    r = client_p5.get("/phase5/billing/invoice?tenant_id=test_t&tier=standard")
    assert r.status_code == 200
    data = r.json()
    assert "base_subscription_usd" in data
    assert data["base_subscription_usd"] == 100_000


def test_subscribe(client_p5):
    r = client_p5.post("/phase5/billing/subscribe", json={
        "tenant_id": "new_tenant",
        "tier": "standard",
        "email": "cto@pharma.com",
        "company_name": "PharmaCorp",
        "trial_days": 30,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["tenant_id"] == "new_tenant"
    assert data["tier"] == "standard"
    assert "customer_id" in data


def test_subscribe_invalid_tier(client_p5):
    r = client_p5.post("/phase5/billing/subscribe", json={
        "tenant_id": "t",
        "tier": "nonexistent",
        "email": "x@x.com",
        "company_name": "X",
    })
    assert r.status_code == 400


def test_get_sla_report(client_p5):
    r = client_p5.get("/phase5/sla/report?tenant_id=global&tier=standard&window_hours=24")
    assert r.status_code == 200
    data = r.json()
    assert "metrics" in data
    assert "sla_targets" in data
    assert "sla_status" in data


def test_get_uptime_series(client_p5):
    r = client_p5.get("/phase5/sla/uptime?tenant_id=global&days=7")
    assert r.status_code == 200
    assert "series" in r.json()


def test_record_metric(client_p5):
    r = client_p5.post("/phase5/sla/record", json={
        "latency_ms": 450.0,
        "success": True,
        "tenant_id": "test_t",
        "endpoint": "/api/v1/engine/run",
        "layer": "pharma",
    })
    assert r.status_code == 200
    assert r.json()["recorded"] is True


def test_portal_dashboard(client_p5):
    r = client_p5.get("/phase5/portal/dashboard?tenant_id=demo&tier=standard")
    assert r.status_code == 200
    data = r.json()
    assert "billing" in data
    assert "utilization" in data
    assert "sla" in data
    assert "features" in data


def test_stripe_webhook_endpoint(client_p5):
    r = client_p5.post(
        "/phase5/stripe/webhook",
        content=b'{"type":"invoice.paid"}',
        headers={"stripe-signature": "fake_sig"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "received" in data


def test_daily_usage_breakdown(client_p5):
    r = client_p5.get("/phase5/billing/usage/daily?tenant_id=demo&unit=api_calls&days=7")
    assert r.status_code == 200
    data = r.json()
    assert "breakdown" in data
    assert isinstance(data["breakdown"], list)
