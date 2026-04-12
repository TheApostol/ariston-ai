"""
Subscription Plans — Phase 5 / Ariston AI.

Pricing tiers aligned with pharma buyer willingness-to-pay.
Per Execution Roadmap: $100K–$500K per customer per trial (Phase 1 target).
"""

from __future__ import annotations

PLANS: dict[str, dict] = {
    "pilot": {
        "name": "Pilot",
        "description": "30-day proof-of-concept. One product, one country.",
        "price_usd_month": 0,
        "price_usd_year": 0,
        "included_units": {
            "api_calls": 500,
            "pipeline_runs": 50,
            "rag_queries": 100,
            "latam_countries": 1,
            "team_seats": 2,
            "audit_retention_years": 1,
        },
        "overage_usd_per_unit": {
            "api_calls": 0.10,
            "pipeline_runs": 5.00,
            "rag_queries": 0.50,
        },
        "features": ["regulatory_intelligence", "latam_1_country", "gxp_audit_basic"],
        "stripe_price_id": None,
    },
    "standard": {
        "name": "Standard",
        "description": "Full LATAM regulatory intelligence. Up to 3 countries.",
        "price_usd_month": 8_333,   # $100K/year
        "price_usd_year": 100_000,
        "included_units": {
            "api_calls": 10_000,
            "pipeline_runs": 500,
            "rag_queries": 2_000,
            "latam_countries": 3,
            "team_seats": 5,
            "audit_retention_years": 5,
        },
        "overage_usd_per_unit": {
            "api_calls": 0.05,
            "pipeline_runs": 2.00,
            "rag_queries": 0.25,
        },
        "features": [
            "regulatory_intelligence", "pharmacovigilance", "csr_generation",
            "clinical_trials_latam", "gxp_audit_full", "webhooks", "api_access",
        ],
        "stripe_price_id": "price_standard_annual",
    },
    "premium": {
        "name": "Premium",
        "description": "Full platform. All LATAM + RWE data licensing.",
        "price_usd_month": 25_000,  # $300K/year
        "price_usd_year": 300_000,
        "included_units": {
            "api_calls": 50_000,
            "pipeline_runs": 2_000,
            "rag_queries": 10_000,
            "latam_countries": 5,
            "team_seats": 20,
            "audit_retention_years": 15,
        },
        "overage_usd_per_unit": {
            "api_calls": 0.02,
            "pipeline_runs": 1.00,
            "rag_queries": 0.10,
        },
        "features": [
            "regulatory_intelligence", "pharmacovigilance", "csr_generation",
            "clinical_trials_latam", "drug_discovery_ai", "biomarker_discovery",
            "rwe_data_licensing", "fda_510k", "international_regulatory",
            "gxp_audit_full", "agent_memory", "webhooks", "api_access",
            "dedicated_support", "sla_99_5",
        ],
        "stripe_price_id": "price_premium_annual",
    },
    "enterprise": {
        "name": "Enterprise",
        "description": "Custom. All phases. Drug discovery AI. Global regulatory.",
        "price_usd_month": None,    # negotiated
        "price_usd_year": None,     # $500K–$2M range
        "included_units": {
            "api_calls": -1,          # unlimited (-1 = unlimited)
            "pipeline_runs": -1,
            "rag_queries": -1,
            "latam_countries": 5,
            "international_countries": -1,
            "team_seats": -1,
            "audit_retention_years": 25,
        },
        "overage_usd_per_unit": {},   # all-inclusive
        "features": [
            "all_platform_features",
            "drug_discovery_ai", "international_regulatory_ema_pmda",
            "custom_rag_integration", "dedicated_infra",
            "gxp_audit_full", "sla_99_9", "white_label_option",
            "on_prem_deployment_option", "custom_integrations",
        ],
        "stripe_price_id": "price_enterprise_custom",
    },
}


def get_plan(tier: str) -> dict:
    return PLANS.get(tier.lower(), PLANS["standard"])


def is_within_quota(tier: str, unit: str, current_usage: int) -> bool:
    plan = get_plan(tier)
    limit = plan["included_units"].get(unit, 0)
    if limit == -1:
        return True  # unlimited
    return current_usage < limit


def overage_cost(tier: str, unit: str, excess_units: int) -> float:
    plan = get_plan(tier)
    rate = plan["overage_usd_per_unit"].get(unit, 0.0)
    return round(rate * excess_units, 2)
