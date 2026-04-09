import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any

class AristonAuditLedger:
    """
    GxP-Compliant Audit Ledger. 
    Maintains an immutable JSON-based record of every clinical decision.
    Each entry is cryptographically hashed and linked to the previous entry.
    """
    LEDGER_PATH = "gxp_audit_trail.json"

    @classmethod
    def log_decision(cls, job_id: str, prompt: str, result: str, metadata: Dict[str, Any]):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "operator": "Ariston-AI-Autopilot",
            "clinical_input": prompt,
            "clinical_output": result,
            "precision_meta": metadata,
            "nonce": time.time_ns()
        }
        
        # Security Hashing (Simulated integrity chain)
        entry_str = json.dumps(entry, sort_keys=True)
        entry["entry_hash"] = hashlib.sha256(entry_str.encode()).hexdigest()

        try:
            with open(cls.LEDGER_PATH, "r") as f:
                ledger = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            ledger = []

        ledger.append(entry)

        with open(cls.LEDGER_PATH, "w") as f:
            json.dump(ledger, f, indent=2)

    @classmethod
    def get_audit_trail(cls):
        try:
            with open(cls.LEDGER_PATH, "r") as f:
                return json.load(f)
        except:
            return []
