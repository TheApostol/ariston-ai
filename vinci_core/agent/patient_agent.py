"""
Longitudinal patient history agent.
Persists patient records in SQLite for cross-session context injection.
"""

import sqlite3
from datetime import datetime
from typing import Optional


class PatientHistoryAgent:
    def __init__(self, db_path: str = "data/ariston.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patient_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    meta TEXT
                )
            """)
            conn.commit()

    def add_record(self, patient_id: str, event_type: str, details: str, date: Optional[str] = None):
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO patient_records (patient_id, event_date, event_type, details) VALUES (?, ?, ?, ?)",
                (patient_id, date, event_type, details),
            )
            conn.commit()

    def get_full_history(self, patient_id: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT event_date, event_type, details FROM patient_records "
                "WHERE patient_id = ? ORDER BY event_date ASC",
                (patient_id,),
            ).fetchall()

        if not rows:
            return ""

        lines = ["--- LONGITUDINAL PATIENT HISTORY ---"]
        for row in rows:
            lines.append(f"[{row[0]}] {row[1].upper()}: {row[2]}")
        lines.append("--- END HISTORY ---")
        return "\n".join(lines)


patient_agent = PatientHistoryAgent()
