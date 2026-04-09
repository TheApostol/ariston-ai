from fastapi import WebSocket
from typing import Dict, List
import json

class ConnectionManager:
    """
    Manages active WebSocket connections for real-time Life Science OS updates.
    """
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def broadcast_job_update(self, job_id: str, status: str, data: dict = None):
        message = json.dumps({
            "job_id": job_id,
            "status": status,
            "data": data or {}
        })
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()
