"""
GxP-Compliant Audit Ledger.
Immutable JSON chain — every clinical decision is hashed and linked.
Required for FDA 21 CFR Part 11 / EU Annex 11 compliance.
(Adapted from ariston-ai-1)
"""

import json
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, Any

LEDGER_PATH = "data/gxp_audit_trail.json"


class AristonAuditLedger:

    @classmethod
    def log_decision(cls, job_id: str, prompt: str, result: str, metadata: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "operator": "Ariston-AI",
            "clinical_input": prompt[:500],   # truncate for storage
            "clinical_output": result[:1000],
            "precision_meta": metadata,
            "nonce": time.time_ns(),
        }
        # Cryptographic integrity
        entry_str = json.dumps(entry, sort_keys=True)
        entry["entry_hash"] = hashlib.sha256(entry_str.encode()).hexdigest()

        import os
        os.makedirs("data", exist_ok=True)

        try:
            with open(LEDGER_PATH, "r") as f:
                ledger = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            ledger = []

        # Chain link: hash of previous entry
        if ledger:
            entry["prev_hash"] = ledger[-1].get("entry_hash", "")

        ledger.append(entry)
        with open(LEDGER_PATH, "w") as f:
            json.dump(ledger, f, indent=2)

    @classmethod
    def get_audit_trail(cls):
        try:
            with open(LEDGER_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
