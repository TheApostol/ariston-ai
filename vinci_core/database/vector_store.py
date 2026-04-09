import sqlite3
import os

class VectorMemoryDB:
    """
    Acts as a persistent RAG memory store. 
    In production, this would be PGVector or ChromaDB backing embedding tensors.
    """
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS conversation_memory 
                            (id INTEGER PRIMARY KEY, prompt TEXT, response TEXT)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS audit_ledger 
                            (id INTEGER PRIMARY KEY, job_id TEXT, timestamp TEXT, entry_hash TEXT, metadata TEXT)''')
                            
    def log_memory(self, prompt: str, response: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO conversation_memory (prompt, response) VALUES (?, ?)", (prompt, response))
        except Exception as e:
            print(f"Memory logging failed: {e}")

    def log_audit_entry(self, job_id: str, timestamp: str, entry_hash: str, metadata: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO audit_ledger (job_id, timestamp, entry_hash, metadata) VALUES (?, ?, ?, ?)", 
                             (job_id, timestamp, entry_hash, metadata))
        except Exception as e:
            print(f"Audit logging failed: {e}")
            
    def get_recent_context(self) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT prompt, response FROM conversation_memory ORDER BY id DESC LIMIT 2").fetchall()
                if not rows:
                    return ""
                return "--- PAST CONTEXT ---\\n" + "\\n".join([f"Prior Prompt: {r[0]}\\nPrior Response: {r[1]}" for r in reversed(rows)]) + "\\n--- END PAST CONTEXT ---\\n"
        except:
            return ""
