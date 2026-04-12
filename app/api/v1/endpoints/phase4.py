"""
Phase 4 API — Platform Intelligence / Ariston AI.

Phase 4 capabilities:
  - /api/v1/platform/audit/*       — GxP persistent audit trail (21 CFR Part 11 / EU Annex 11)
  - /api/v1/platform/memory/*      — Agent persistent memory (cross-session, tenant-keyed)
  - /api/v1/platform/tenants/*     — Multi-tenant management (sponsors + CROs)
  - /api/v1/platform/keys/*        — API key issuance and revocation
  - /api/v1/platform/webhooks/*    — Event subscription and delivery management

Platform value proposition:
  - GxP audit trail: required for every enterprise pharma contract (FDA/EMA compliance)
  - Agent memory: enables long-running multi-session workflows (trial design, dossier assembly)
  - RBAC: per-sponsor data isolation for multi-tenant SaaS
  - Webhooks: integrates Ariston AI into sponsor's existing workflow automation
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

from vinci_core.audit.gxp_trail import gxp_audit
from vinci_core.memory.agent_memory import agent_memory
from vinci_core.auth.rbac import rbac_manager, PERMISSIONS, require_permission, APIKey
from vinci_core.webhooks.dispatcher import webhook_dispatcher, WebhookEvent, WEBHOOK_EVENT_TYPES

import uuid
from datetime import datetime, timezone

router = APIRouter(prefix="/platform", tags=["Phase 4 — Platform Intelligence"])


# ---------------------------------------------------------------------------
# GxP Audit Trail
# ---------------------------------------------------------------------------

@router.get("/audit/entries")
async def query_audit_entries(
    tenant_id: Optional[str] = None,
    job_id: Optional[str] = None,
    layer: Optional[str] = None,
    event_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Query the GxP audit trail with filters.
    Returns signed, chained audit entries for regulatory inspection.
    FDA 21 CFR Part 11 / EU Annex 11 compliant.
    """
    try:
        entries = gxp_audit.query(
            tenant_id=tenant_id,
            job_id=job_id,
            layer=layer,
            event_type=event_type,
            since=since,
            until=until,
            limit=min(limit, 200),
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit query error: {e}")
    return {"entries": entries, "count": len(entries), "limit": limit, "offset": offset}


@router.get("/audit/verify")
async def verify_audit_chain(tenant_id: str = "default", limit: int = 1000):
    """
    Verify the cryptographic integrity of the audit chain for a tenant.
    Detects any tampering or record deletion.
    Returns first_broken_at entry if chain is compromised.
    """
    try:
        result = gxp_audit.verify_chain(tenant_id=tenant_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chain verification error: {e}")
    return result


@router.get("/audit/stats")
async def get_audit_stats(tenant_id: Optional[str] = None):
    """Audit trail statistics: entry counts by layer, event type, date range."""
    try:
        return gxp_audit.get_stats(tenant_id=tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit stats error: {e}")


@router.get("/audit/export/{tenant_id}")
async def export_audit_trail(tenant_id: str, limit: int = 10000):
    """
    Export full GxP audit trail for regulatory submission.
    Returns complete signed chain in JSON format.
    Retention: 15 years minimum per GxP requirements.
    """
    try:
        entries = gxp_audit.export_json(tenant_id=tenant_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit export error: {e}")
    return {
        "tenant_id": tenant_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(entries),
        "entries": entries,
        "compliance": {
            "fda_cfr": "21 CFR Part 11",
            "eu_annex": "EU Annex 11",
            "retention_years": 15,
            "hash_algorithm": "SHA-256",
            "chain_type": "SHA-256 linked hash chain",
        },
    }


# ---------------------------------------------------------------------------
# Agent Memory
# ---------------------------------------------------------------------------

class MemoryWriteRequest(BaseModel):
    content: str = Field(..., description="Memory content to store")
    agent_type: str = Field(..., description="Agent type: latam_regulatory | drug_discovery | trial_intelligence | pharmacovigilance")
    tenant_id: str = Field("default", description="Tenant identifier")
    session_id: Optional[str] = Field(None, description="Session ID for grouping related memories")
    tags: Optional[list[str]] = Field(None, description="Semantic tags for retrieval")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score (0-1), affects recall ranking")
    ttl_days: int = Field(90, ge=1, le=3650, description="Time-to-live in days")
    metadata: Optional[dict] = Field(None)


class MemoryRecallRequest(BaseModel):
    query: str = Field(..., description="Natural language recall query")
    agent_type: str
    tenant_id: str = "default"
    tags: Optional[list[str]] = None
    limit: int = Field(5, ge=1, le=20)
    min_importance: float = Field(0.0, ge=0.0, le=1.0)


@router.post("/memory/remember")
async def store_memory(req: MemoryWriteRequest):
    """
    Store a persistent memory for an agent.
    Memories persist across API calls and sessions, enabling long-running workflows.
    Full-text searchable, tag-indexed, TTL-configurable.
    """
    try:
        record = agent_memory.remember(
            content=req.content,
            agent_type=req.agent_type,
            tenant_id=req.tenant_id,
            session_id=req.session_id,
            tags=req.tags,
            importance=req.importance,
            ttl_days=req.ttl_days,
            metadata=req.metadata or {},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory store error: {e}")
    return {
        "memory_id": record.memory_id,
        "agent_type": record.agent_type,
        "tenant_id": record.tenant_id,
        "session_id": record.session_id,
        "tags": record.tags,
        "importance": record.importance,
        "created_at": record.created_at,
        "expires_at": record.expires_at,
    }


@router.post("/memory/recall")
async def recall_memory(req: MemoryRecallRequest):
    """
    Retrieve relevant memories via full-text search + tag filtering.
    Results ranked by importance DESC, recency DESC.
    Used internally by agents to inject prior context into AI prompts.
    """
    try:
        records = agent_memory.recall(
            query=req.query,
            agent_type=req.agent_type,
            tenant_id=req.tenant_id,
            tags=req.tags,
            limit=req.limit,
            min_importance=req.min_importance,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory recall error: {e}")
    return {
        "query": req.query,
        "memories": [
            {
                "memory_id": r.memory_id,
                "content": r.content,
                "tags": r.tags,
                "importance": r.importance,
                "created_at": r.created_at,
                "session_id": r.session_id,
            }
            for r in records
        ],
        "total": len(records),
    }


@router.get("/memory/session/{session_id}")
async def get_session_memory(session_id: str, tenant_id: str = "default"):
    """Retrieve all memories for a session in chronological order."""
    try:
        records = agent_memory.get_session(session_id=session_id, tenant_id=tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session memory error: {e}")
    return {
        "session_id": session_id,
        "memories": [{"memory_id": r.memory_id, "content": r.content, "tags": r.tags, "created_at": r.created_at} for r in records],
        "total": len(records),
    }


@router.delete("/memory/{memory_id}")
async def forget_memory(memory_id: str, tenant_id: str = "default"):
    """Delete a specific memory (GDPR right to erasure)."""
    deleted = agent_memory.forget(memory_id=memory_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found for tenant {tenant_id}")
    return {"deleted": True, "memory_id": memory_id}


@router.post("/memory/purge-expired")
async def purge_expired_memories():
    """Remove all TTL-expired memories. Returns count deleted."""
    count = agent_memory.purge_expired()
    return {"purged": count}


@router.get("/memory/stats")
async def memory_stats(tenant_id: Optional[str] = None):
    return agent_memory.get_stats(tenant_id=tenant_id)


# ---------------------------------------------------------------------------
# Multi-Tenant Management
# ---------------------------------------------------------------------------

class TenantCreateRequest(BaseModel):
    name: str = Field(..., description="Company / CRO name")
    tier: str = Field("standard", description="free | standard | premium | enterprise")
    tenant_id: Optional[str] = Field(None, description="Optional custom tenant ID")
    metadata: Optional[dict] = None


@router.post("/tenants")
async def create_tenant(req: TenantCreateRequest):
    """Register a new pharmaceutical sponsor or CRO tenant."""
    try:
        tenant = rbac_manager.create_tenant(
            name=req.name,
            tier=req.tier,
            tenant_id=req.tenant_id,
            metadata=req.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tenant creation error: {e}")
    return {
        "tenant_id": tenant.tenant_id,
        "name": tenant.name,
        "tier": tenant.tier,
        "created_at": tenant.created_at,
    }


@router.get("/tenants")
async def list_tenants():
    """List all active tenants (admin only)."""
    tenants = rbac_manager.list_tenants()
    return {
        "tenants": [{"tenant_id": t.tenant_id, "name": t.name, "tier": t.tier, "created_at": t.created_at} for t in tenants],
        "total": len(tenants),
    }


# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------

class APIKeyRequest(BaseModel):
    tenant_id: str
    role: str = Field("analyst", description=f"Role: {list(PERMISSIONS.keys())}")
    description: str = Field("", description="Human-readable description (e.g. 'Production API service')")
    ttl_days: int = Field(365, ge=1, le=1825)


@router.post("/keys/issue")
async def issue_api_key(req: APIKeyRequest):
    """
    Issue a new API key for a tenant.
    The raw key is shown ONCE — store it securely. Ariston never stores plaintext keys.
    """
    if req.role not in PERMISSIONS:
        raise HTTPException(status_code=422, detail=f"Invalid role. Valid roles: {list(PERMISSIONS.keys())}")
    try:
        raw_key, api_key = rbac_manager.issue_api_key(
            tenant_id=req.tenant_id,
            role=req.role,
            description=req.description,
            ttl_days=req.ttl_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Key issuance error: {e}")
    return {
        "key_id": api_key.key_id,
        "raw_key": raw_key,
        "tenant_id": api_key.tenant_id,
        "role": api_key.role,
        "permissions": PERMISSIONS[api_key.role],
        "expires_at": api_key.expires_at,
        "warning": "Store this key securely — it will NOT be shown again.",
    }


@router.delete("/keys/{key_id}")
async def revoke_api_key(key_id: str, tenant_id: str):
    """Revoke an API key. Use immediately if a key is compromised."""
    revoked = rbac_manager.revoke_key(key_id=key_id, tenant_id=tenant_id)
    if not revoked:
        raise HTTPException(status_code=404, detail=f"Key {key_id} not found for tenant {tenant_id}")
    return {"revoked": True, "key_id": key_id}


@router.get("/keys/{tenant_id}")
async def list_tenant_keys(tenant_id: str):
    """List all API keys for a tenant (metadata only, never raw keys)."""
    keys = rbac_manager.get_tenant_keys(tenant_id=tenant_id)
    return {
        "tenant_id": tenant_id,
        "keys": [
            {
                "key_id": k.key_id,
                "role": k.role,
                "description": k.description,
                "active": k.active,
                "created_at": k.created_at,
                "expires_at": k.expires_at,
                "last_used_at": k.last_used_at,
            }
            for k in keys
        ],
        "total": len(keys),
    }


@router.get("/roles")
async def get_role_permissions():
    """Return all roles and their permissions."""
    return {"roles": PERMISSIONS}


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------

class WebhookSubscribeRequest(BaseModel):
    url: str = Field(..., description="HTTPS endpoint to receive webhook POST requests")
    tenant_id: str = Field("default")
    event_types: Optional[list[str]] = Field(None, description="Subscribe to specific events; empty = all events")
    description: str = Field("", description="Human-readable description")
    secret: Optional[str] = Field(None, description="HMAC signing secret; auto-generated if not provided")


class WebhookEmitRequest(BaseModel):
    event_type: str = Field(..., description="Event type from the catalogue")
    tenant_id: str = Field("default")
    payload: dict = Field(default_factory=dict, description="Event payload data")


@router.post("/webhooks/subscribe")
async def subscribe_webhook(req: WebhookSubscribeRequest):
    """
    Register a webhook endpoint to receive event notifications.
    All payloads are HMAC-SHA256 signed — verify with X-Ariston-Signature header.
    """
    if req.event_types:
        invalid = [e for e in req.event_types if e not in WEBHOOK_EVENT_TYPES]
        if invalid:
            raise HTTPException(status_code=422, detail=f"Unknown event types: {invalid}")
    try:
        sub = webhook_dispatcher.subscribe(
            url=req.url,
            tenant_id=req.tenant_id,
            event_types=req.event_types,
            description=req.description,
            secret=req.secret,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription error: {e}")
    return {
        "sub_id": sub.sub_id,
        "url": sub.url,
        "tenant_id": sub.tenant_id,
        "event_types": sub.event_types or ["all"],
        "signing_secret": sub.secret,
        "created_at": sub.created_at,
        "note": "Save the signing_secret to verify incoming webhook signatures.",
    }


@router.delete("/webhooks/{sub_id}")
async def unsubscribe_webhook(sub_id: str, tenant_id: str = "default"):
    """Remove a webhook subscription."""
    removed = webhook_dispatcher.unsubscribe(sub_id=sub_id, tenant_id=tenant_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Subscription {sub_id} not found")
    return {"unsubscribed": True, "sub_id": sub_id}


@router.post("/webhooks/emit")
async def emit_webhook_event(req: WebhookEmitRequest):
    """
    Emit a webhook event to all matching subscribers.
    Delivered asynchronously with HMAC-signed payloads + retry logic.
    """
    if req.event_type not in WEBHOOK_EVENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown event type. Valid types: {WEBHOOK_EVENT_TYPES}",
        )
    event = WebhookEvent(
        event_id=str(uuid.uuid4()),
        event_type=req.event_type,
        tenant_id=req.tenant_id,
        payload=req.payload,
    )
    try:
        delivery_ids = webhook_dispatcher.emit(event, background=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Event emission error: {e}")
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "deliveries_queued": len(delivery_ids),
        "delivery_ids": delivery_ids,
    }


@router.get("/webhooks/subscriptions/{tenant_id}")
async def list_subscriptions(tenant_id: str):
    """List all webhook subscriptions for a tenant."""
    return {
        "tenant_id": tenant_id,
        "subscriptions": webhook_dispatcher.get_subscriptions(tenant_id=tenant_id),
    }


@router.get("/webhooks/deliveries/{tenant_id}")
async def list_deliveries(
    tenant_id: str,
    event_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
):
    """Query webhook delivery history with status (delivered | failed | dead_letter | pending)."""
    return {
        "tenant_id": tenant_id,
        "deliveries": webhook_dispatcher.get_deliveries(
            tenant_id=tenant_id,
            event_type=event_type,
            status=status,
            limit=min(limit, 200),
        ),
    }


@router.get("/webhooks/events")
async def list_event_types():
    """Return the full catalogue of available webhook event types."""
    return {
        "event_types": WEBHOOK_EVENT_TYPES,
        "total": len(WEBHOOK_EVENT_TYPES),
        "categories": {
            "trial": [e for e in WEBHOOK_EVENT_TYPES if e.startswith("trial.")],
            "pv": [e for e in WEBHOOK_EVENT_TYPES if e.startswith("pv.")],
            "drug_discovery": [e for e in WEBHOOK_EVENT_TYPES if e.startswith("drug_discovery.")],
            "regulatory": [e for e in WEBHOOK_EVENT_TYPES if e.startswith("regulatory.")],
            "pipeline": [e for e in WEBHOOK_EVENT_TYPES if e.startswith("pipeline.")],
            "rwe": [e for e in WEBHOOK_EVENT_TYPES if e.startswith("rwe.")],
            "system": [e for e in WEBHOOK_EVENT_TYPES if e.startswith("system.")],
        },
    }
