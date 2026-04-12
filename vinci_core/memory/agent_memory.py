"""
Agent Memory Store — Phase 4 / Ariston AI.

Persistent, tenant-keyed cross-session memory for AI agents.
Enables agents to recall prior interactions, accumulated knowledge,
and task state across API calls — critical for multi-step workflows.

Use cases:
  - Regulatory agent recalls prior country submissions for a sponsor
  - Clinical trial agent tracks protocol amendments across sessions
  - Pharmacovigilance agent correlates AEs across reporting periods
  - Drug discovery agent accumulates validated/rejected hypotheses

Architecture:
  - SQLite backend (upgradeable to Redis or pgvector in production)
  - Semantic tag indexing for fast retrieval by topic
  - TTL-based expiry for session-scoped memories
  - Per-tenant isolation (multi-tenant safe)
  - Configurable memory window (last N memories per agent)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

logger = logging.getLogger("ariston.memory")

_DB_PATH = os.environ.get("ARISTON_MEMORY_DB", "data/agent_memory.db")
_DEFAULT_TTL_DAYS = 90


@dataclass
class MemoryRecord:
    """A single memory entry for an agent."""
    memory_id: str
    tenant_id: str
    agent_type: str          # latam_regulatory | drug_discovery | trial_intelligence | pharmacovigilance
    session_id: str
    content: str
    tags: list[str]
    importance: float        # 0.0–1.0; higher = retrieved first
    created_at: str
    expires_at: str
    metadata: dict = field(default_factory=dict)


class AgentMemoryStore:
    """
    Persistent per-tenant agent memory.

    Provides:
    - remember()       — store a memory for an agent/tenant
    - recall()         — retrieve relevant memories by tag or keyword
    - forget()         — delete a specific memory (GDPR compliance)
    - purge_expired()  — clean up TTL-expired entries
    - get_session()    — retrieve full session memory
    - summarize()      — aggregate memory window into a context string
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
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
                CREATE TABLE IF NOT EXISTS memories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id   TEXT UNIQUE NOT NULL,
                    tenant_id   TEXT NOT NULL,
                    agent_type  TEXT NOT NULL,
                    session_id  TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    tags_json   TEXT NOT NULL DEFAULT '[]',
                    importance  REAL NOT NULL DEFAULT 0.5,
                    created_at  TEXT NOT NULL,
                    expires_at  TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_mem_tenant  ON memories(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_mem_agent   ON memories(agent_type);
                CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session_id);
                CREATE INDEX IF NOT EXISTS idx_mem_expires ON memories(expires_at);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(memory_id UNINDEXED, content, tags_json,
                           content='memories', content_rowid='id');

                CREATE TRIGGER IF NOT EXISTS mem_fts_insert
                AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, memory_id, content, tags_json)
                    VALUES (new.id, new.memory_id, new.content, new.tags_json);
                END;

                CREATE TRIGGER IF NOT EXISTS mem_fts_delete
                AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, memory_id, content, tags_json)
                    VALUES ('delete', old.id, old.memory_id, old.content, old.tags_json);
                END;
            """)

    def remember(
        self,
        content: str,
        agent_type: str,
        tenant_id: str = "default",
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
        ttl_days: int = _DEFAULT_TTL_DAYS,
        metadata: Optional[dict] = None,
    ) -> MemoryRecord:
        """Store a new memory for an agent."""
        now = datetime.now(timezone.utc)
        record = MemoryRecord(
            memory_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            agent_type=agent_type,
            session_id=session_id or str(uuid.uuid4()),
            content=content,
            tags=tags or [],
            importance=max(0.0, min(1.0, importance)),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=ttl_days)).isoformat(),
            metadata=metadata or {},
        )

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO memories
                  (memory_id, tenant_id, agent_type, session_id, content,
                   tags_json, importance, created_at, expires_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.memory_id, record.tenant_id, record.agent_type,
                record.session_id, record.content,
                json.dumps(record.tags), record.importance,
                record.created_at, record.expires_at,
                json.dumps(record.metadata),
            ))

        logger.debug("[Memory] stored memory_id=%s agent=%s tenant=%s", record.memory_id, agent_type, tenant_id)
        return record

    def recall(
        self,
        query: str,
        agent_type: str,
        tenant_id: str = "default",
        tags: Optional[list[str]] = None,
        limit: int = 5,
        min_importance: float = 0.0,
    ) -> list[MemoryRecord]:
        """
        Retrieve relevant memories via full-text search + tag filtering.
        Results ranked by importance DESC, recency DESC.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            # FTS search
            try:
                fts_rows = conn.execute("""
                    SELECT m.* FROM memories m
                    INNER JOIN memories_fts fts ON m.id = fts.rowid
                    WHERE fts.content MATCH ?
                      AND m.tenant_id = ?
                      AND m.agent_type = ?
                      AND m.expires_at > ?
                      AND m.importance >= ?
                    ORDER BY m.importance DESC, m.created_at DESC
                    LIMIT ?
                """, (query, tenant_id, agent_type, now, min_importance, limit)).fetchall()
            except Exception:
                fts_rows = []

            # Tag-filtered fallback / supplement
            if tags:
                tag_rows = conn.execute("""
                    SELECT * FROM memories
                    WHERE tenant_id = ?
                      AND agent_type = ?
                      AND expires_at > ?
                      AND importance >= ?
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                """, (tenant_id, agent_type, now, min_importance, limit * 2)).fetchall()
                # Filter by tags in Python
                tag_set = set(t.lower() for t in tags)
                tagged = [
                    r for r in tag_rows
                    if tag_set & set(t.lower() for t in json.loads(r["tags_json"] or "[]"))
                ]
                # Merge FTS + tag results, deduplicate
                seen = {r["memory_id"] for r in fts_rows}
                merged = list(fts_rows) + [r for r in tagged if r["memory_id"] not in seen]
                merged = sorted(merged, key=lambda r: (-r["importance"], r["created_at"]))
                rows = merged[:limit]
            else:
                rows = fts_rows[:limit]

        return [self._row_to_record(r) for r in rows]

    def get_session(self, session_id: str, tenant_id: str = "default") -> list[MemoryRecord]:
        """Retrieve all memories for a session, ordered chronologically."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM memories
                WHERE session_id = ? AND tenant_id = ? AND expires_at > ?
                ORDER BY created_at ASC
            """, (session_id, tenant_id, now)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def summarize(
        self,
        agent_type: str,
        tenant_id: str = "default",
        query: Optional[str] = None,
        limit: int = 8,
    ) -> str:
        """
        Assemble a memory context string for LLM injection.
        Returns empty string if no relevant memories found.
        """
        if query:
            memories = self.recall(query, agent_type, tenant_id, limit=limit)
        else:
            now = datetime.now(timezone.utc).isoformat()
            with self._conn() as conn:
                rows = conn.execute("""
                    SELECT * FROM memories
                    WHERE tenant_id = ? AND agent_type = ? AND expires_at > ?
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                """, (tenant_id, agent_type, now, limit)).fetchall()
            memories = [self._row_to_record(r) for r in rows]

        if not memories:
            return ""

        lines = ["AGENT MEMORY CONTEXT (prior interactions):"]
        for m in memories:
            ts = m.created_at[:19].replace("T", " ")
            tags_str = f" [{', '.join(m.tags)}]" if m.tags else ""
            lines.append(f"[{ts}{tags_str}] {m.content}")
        return "\n".join(lines)

    def forget(self, memory_id: str, tenant_id: str = "default") -> bool:
        """Delete a specific memory (GDPR right to erasure)."""
        with self._conn() as conn:
            result = conn.execute(
                "DELETE FROM memories WHERE memory_id = ? AND tenant_id = ?",
                (memory_id, tenant_id),
            )
        deleted = result.rowcount > 0
        if deleted:
            logger.info("[Memory] deleted memory_id=%s tenant=%s", memory_id, tenant_id)
        return deleted

    def purge_expired(self) -> int:
        """Remove all expired memories. Returns count deleted."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            result = conn.execute("DELETE FROM memories WHERE expires_at <= ?", (now,))
        count = result.rowcount
        if count:
            logger.info("[Memory] purged %d expired memories", count)
        return count

    def get_stats(self, tenant_id: Optional[str] = None) -> dict:
        clause = "WHERE tenant_id = ?" if tenant_id else ""
        params = [tenant_id] if tenant_id else []
        with self._conn() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM memories {clause}", params).fetchone()[0]
            by_agent = conn.execute(
                f"SELECT agent_type, COUNT(*) as cnt FROM memories {clause} GROUP BY agent_type", params
            ).fetchall()
        return {
            "tenant_id": tenant_id or "all",
            "total_memories": total,
            "by_agent_type": {r["agent_type"]: r["cnt"] for r in by_agent},
        }

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            tenant_id=row["tenant_id"],
            agent_type=row["agent_type"],
            session_id=row["session_id"],
            content=row["content"],
            tags=json.loads(row["tags_json"] or "[]"),
            importance=row["importance"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )


agent_memory = AgentMemoryStore()
