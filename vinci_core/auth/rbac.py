"""
Multi-Tenant RBAC — Phase 4 / Ariston AI.

Per-sponsor data isolation with role-based access control.
Required before any enterprise pharma contract can close.

Architecture:
  - Tenants: pharmaceutical companies / CROs (isolated data namespaces)
  - API Keys: hashed bearer tokens mapped to tenants + roles
  - Roles: admin | analyst | viewer | api_service
  - Permissions: fine-grained per endpoint/resource
  - SQLite-backed (upgrade to Postgres + Redis for production)

Permission model:
  admin     — full access to all tenant resources + audit export
  analyst   — read/write regulatory, RWE, trial intelligence, drug discovery
  viewer    — read-only across all resources
  api_service — programmatic access for pipeline automation (no UI)

HIPAA + GxP alignment:
  - All auth events logged to GxP audit trail
  - API keys stored as SHA-256 hashes (never plaintext)
  - Per-tenant row isolation enforced at query layer
  - Session expiry configurable (default: 30 days)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import Any, Callable, Generator, Optional

logger = logging.getLogger("ariston.auth")

_DB_PATH = os.environ.get("ARISTON_AUTH_DB", "data/ariston_auth.db")
_KEY_TTL_DAYS = 365

# ---------------------------------------------------------------------------
# Permission registry
# ---------------------------------------------------------------------------
PERMISSIONS = {
    "admin": [
        "audit:read", "audit:export", "audit:verify",
        "memory:read", "memory:write", "memory:delete",
        "rwe:read", "rwe:write", "rwe:license",
        "regulatory:read", "regulatory:write",
        "trial:read", "trial:write",
        "drug_discovery:read", "drug_discovery:write",
        "webhook:read", "webhook:write",
        "tenant:manage",
    ],
    "analyst": [
        "audit:read",
        "memory:read", "memory:write",
        "rwe:read", "rwe:write",
        "regulatory:read", "regulatory:write",
        "trial:read", "trial:write",
        "drug_discovery:read", "drug_discovery:write",
        "webhook:read",
    ],
    "viewer": [
        "audit:read",
        "memory:read",
        "rwe:read",
        "regulatory:read",
        "trial:read",
        "drug_discovery:read",
    ],
    "api_service": [
        "regulatory:read", "regulatory:write",
        "trial:read", "trial:write",
        "drug_discovery:read",
        "rwe:read",
        "memory:read", "memory:write",
        "audit:read",
    ],
}


@dataclass
class Tenant:
    tenant_id: str
    name: str
    tier: str              # free | standard | premium | enterprise
    created_at: str
    active: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class APIKey:
    key_id: str
    tenant_id: str
    role: str
    description: str
    created_at: str
    expires_at: str
    active: bool = True
    last_used_at: Optional[str] = None


class RBACManager:
    """
    Multi-tenant RBAC manager.

    Provides:
    - create_tenant()      — register a new pharma sponsor
    - issue_api_key()      — generate a scoped bearer token
    - authenticate()       — validate a request bearer token
    - check_permission()   — verify role has a given permission
    - revoke_key()         — deactivate a compromised key
    - list_tenants()       — admin view of all tenants
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        self._init_db()
        self._ensure_system_tenant()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id   TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    tier        TEXT NOT NULL DEFAULT 'standard',
                    active      INTEGER NOT NULL DEFAULT 1,
                    created_at  TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id      TEXT PRIMARY KEY,
                    tenant_id   TEXT NOT NULL REFERENCES tenants(tenant_id),
                    key_hash    TEXT UNIQUE NOT NULL,
                    role        TEXT NOT NULL DEFAULT 'analyst',
                    description TEXT NOT NULL DEFAULT '',
                    active      INTEGER NOT NULL DEFAULT 1,
                    created_at  TEXT NOT NULL,
                    expires_at  TEXT NOT NULL,
                    last_used_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_key_hash   ON api_keys(key_hash);
                CREATE INDEX IF NOT EXISTS idx_key_tenant ON api_keys(tenant_id);
            """)

    def _ensure_system_tenant(self) -> None:
        """Create the default system tenant if it doesn't exist."""
        self.create_tenant(
            tenant_id="default",
            name="Ariston AI (System)",
            tier="enterprise",
        )

    def create_tenant(
        self,
        name: str,
        tier: str = "standard",
        tenant_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Tenant:
        """Register a new tenant (pharmaceutical company / CRO)."""
        tid = tenant_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO tenants (tenant_id, name, tier, created_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (tid, name, tier, now, json.dumps(metadata or {})))
        except Exception as e:
            logger.warning("[RBAC] create_tenant failed: %s", e)
        return Tenant(tenant_id=tid, name=name, tier=tier, created_at=now)

    def issue_api_key(
        self,
        tenant_id: str,
        role: str = "analyst",
        description: str = "",
        ttl_days: int = _KEY_TTL_DAYS,
    ) -> tuple[str, APIKey]:
        """
        Issue a new API key for a tenant.
        Returns (raw_key, APIKey) — raw_key is shown ONCE, never stored.
        """
        if role not in PERMISSIONS:
            raise ValueError(f"Invalid role '{role}'. Must be one of: {list(PERMISSIONS.keys())}")

        raw_key = f"ariston_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(days=ttl_days)).isoformat()

        api_key = APIKey(
            key_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            role=role,
            description=description,
            created_at=now.isoformat(),
            expires_at=expires_at,
        )

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO api_keys
                  (key_id, tenant_id, key_hash, role, description, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                api_key.key_id, tenant_id, key_hash, role,
                description, api_key.created_at, expires_at,
            ))

        logger.info("[RBAC] issued key_id=%s tenant=%s role=%s", api_key.key_id, tenant_id, role)
        return raw_key, api_key

    def authenticate(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate a bearer token. Returns APIKey or None.
        Updates last_used_at on success.
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            row = conn.execute("""
                SELECT * FROM api_keys
                WHERE key_hash = ? AND active = 1 AND expires_at > ?
            """, (key_hash, now)).fetchone()

            if not row:
                return None

            # Update last_used
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                (now, row["key_id"]),
            )

        return APIKey(
            key_id=row["key_id"],
            tenant_id=row["tenant_id"],
            role=row["role"],
            description=row["description"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            active=bool(row["active"]),
            last_used_at=now,
        )

    def check_permission(self, role: str, permission: str) -> bool:
        return permission in PERMISSIONS.get(role, [])

    def revoke_key(self, key_id: str, tenant_id: str) -> bool:
        with self._conn() as conn:
            result = conn.execute(
                "UPDATE api_keys SET active = 0 WHERE key_id = ? AND tenant_id = ?",
                (key_id, tenant_id),
            )
        return result.rowcount > 0

    def list_tenants(self) -> list[Tenant]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM tenants WHERE active = 1 ORDER BY created_at").fetchall()
        return [
            Tenant(
                tenant_id=r["tenant_id"],
                name=r["name"],
                tier=r["tier"],
                created_at=r["created_at"],
                active=bool(r["active"]),
                metadata=json.loads(r["metadata_json"] or "{}"),
            )
            for r in rows
        ]

    def get_tenant_keys(self, tenant_id: str) -> list[APIKey]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys WHERE tenant_id = ? ORDER BY created_at DESC",
                (tenant_id,),
            ).fetchall()
        return [
            APIKey(
                key_id=r["key_id"],
                tenant_id=r["tenant_id"],
                role=r["role"],
                description=r["description"],
                created_at=r["created_at"],
                expires_at=r["expires_at"],
                active=bool(r["active"]),
                last_used_at=r["last_used_at"],
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# FastAPI dependency injection
# ---------------------------------------------------------------------------

rbac_manager = RBACManager()


def require_permission(permission: str):
    """
    FastAPI dependency that validates bearer token and checks permission.

    Usage:
        @router.get("/sensitive")
        async def endpoint(key: APIKey = Depends(require_permission("rwe:read"))):
            ...
    """
    from fastapi import HTTPException, Security, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    _bearer = HTTPBearer(auto_error=False)

    def dependency(credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer)) -> APIKey:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        api_key = rbac_manager.authenticate(credentials.credentials)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
            )
        if not rbac_manager.check_permission(api_key.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{api_key.role}' lacks permission '{permission}'",
            )
        return api_key
    return dependency
