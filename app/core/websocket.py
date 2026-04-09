"""
WebSocket connection manager for real-time job status streaming.
"""

import json
from fastapi import WebSocket
from typing import Dict


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send(self, client_id: str, message: dict):
        ws = self.active_connections.get(client_id)
        if ws:
            await ws.send_text(json.dumps(message))

    async def broadcast(self, job_id: str, status: str, data: dict = None):
        message = json.dumps({"job_id": job_id, "status": status, "data": data or {}})
        for ws in list(self.active_connections.values()):
            try:
                await ws.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()
