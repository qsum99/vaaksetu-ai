from fastapi import APIRouter, HTTPException
import uuid
from typing import Dict
from backend.models.session import SessionState

router = APIRouter(tags=["Session"])

# In-memory store for sessions (for simplicity, we could use Redis)
active_sessions: Dict[str, SessionState] = {}

@router.post("/start", response_model=SessionState)
async def start_session() -> SessionState:
    session_id = str(uuid.uuid4())
    session = SessionState(session_id=session_id)
    active_sessions[session_id] = session
    return session

@router.get("/{session_id}", response_model=SessionState)
async def get_session(session_id: str) -> SessionState:
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return active_sessions[session_id]
