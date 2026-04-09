import sqlite3
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict

# Standard Console Logger for Internal Engine Observability
vinci_logger = logging.getLogger("vinci")
vinci_logger.setLevel(logging.INFO)
if not vinci_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    vinci_logger.addHandler(handler)

class ClinicalAuditLogger:
    """
    Implements a structured audit trail for Life Science compliance.
    Records every AI orchestration event to a persistent database.
    """
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    intent TEXT,
                    model TEXT,
                    prompt_preview TEXT,
                    response_preview TEXT,
                    consensus_score REAL,
                    safety_flag INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            conn.commit()

    def log_event(
        self, 
        job_id: str, 
        intent: str, 
        model: str, 
        prompt: str, 
        response: str = "", 
        score: float = 0.0, 
        safety_violation: bool = False,
        meta: Dict[str, Any] = None
    ):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO audit_logs (job_id, timestamp, intent, model, prompt_preview, response_preview, consensus_score, safety_flag, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        job_id,
                        datetime.now().isoformat(),
                        intent,
                        model,
                        prompt[:200], # Store preview for audit
                        response[:500],
                        score,
                        1 if safety_violation else 0,
                        json.dumps(meta or {})
                    )
                )
                conn.commit()
        except Exception as e:
            logging.error(f"Audit Logging Failed: {e}")

audit_logger = ClinicalAuditLogger()
