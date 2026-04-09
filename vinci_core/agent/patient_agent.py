import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

class PatientHistoryAgent:
    """
    Manages longitudinal patient records, including clinical history,
    symptoms per date, and historical diagnostic analysis.
    """
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS patient_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    meta TEXT
                )
            ''')
            conn.commit()

    def add_record(self, patient_id: str, event_type: str, details: str, date: str = None):
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO patient_records (patient_id, event_date, event_type, details) VALUES (?, ?, ?, ?)",
                    (patient_id, date, event_type, details)
                )
                conn.commit()
        except Exception as e:
            print(f"Failed to add patient record: {e}")

    def get_full_history(self, patient_id: str) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT event_date, event_type, details FROM patient_records WHERE patient_id = ? ORDER BY event_date ASC",
                    (patient_id,)
                ).fetchall()
                
                if not rows:
                    return "No historical records found for this patient."
                
                history_text = "--- LONGITUDINAL PATIENT HISTORY ---\n"
                for row in rows:
                    history_text += f"[{row[0]}] {row[1].upper()}: {row[2]}\n"
                history_text += "--- END HISTORY ---"
                return history_text
        except Exception as e:
            return f"Error retrieving patient history: {str(e)}"

# Singleton for engine use
patient_agent = PatientHistoryAgent()
