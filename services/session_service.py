"""Session service — business logic for session management."""
import uuid

# In-memory store (shared with routes via import)
_sessions = {}


def create_session():
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "context": [],
        "transcript": "",
        "detected_languages": [],
        "extracted_fields": {},
        "detected_symptoms": [],
    }
    _sessions[session_id] = session
    return session


def get_session(session_id):
    return _sessions.get(session_id)


def update_session(session_id, updates):
    if session_id in _sessions:
        _sessions[session_id].update(updates)
        return _sessions[session_id]
    return None


def get_or_create_session(session_id):
    if session_id not in _sessions:
        _sessions[session_id] = {
            "session_id": session_id,
            "context": [],
            "transcript": "",
            "detected_languages": [],
            "extracted_fields": {},
            "detected_symptoms": [],
        }
    return _sessions[session_id]
