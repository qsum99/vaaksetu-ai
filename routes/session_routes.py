"""Session routes — in-memory session management."""
import uuid
from flask import Blueprint, jsonify

session_bp = Blueprint("session", __name__, url_prefix="/api/session")

# In-memory session store
active_sessions = {}


def _new_session(session_id):
    return {
        "session_id": session_id,
        "context": [],
        "transcript": "",
        "detected_languages": [],
        "extracted_fields": {},
        "detected_symptoms": [],
    }


@session_bp.route("/start", methods=["POST"])
def start_session():
    session_id = str(uuid.uuid4())
    session = _new_session(session_id)
    active_sessions[session_id] = session
    return jsonify(session), 201


@session_bp.route("/<session_id>", methods=["GET"])
def get_session(session_id):
    session = active_sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session)
