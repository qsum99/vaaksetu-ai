"""
Session Routes — Conversation session management with DB persistence.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

session_bp = Blueprint("session", __name__, url_prefix="/api/session")

# ---------------------------------------------------------------------------
# In-memory active session store (live ConversationManager instances)
# ---------------------------------------------------------------------------
_active_sessions: dict = {}


def _get_manager(session_id: str = None):
    """Get existing or create new ConversationManager."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ai_core.llm import ConversationManager

    if session_id and session_id in _active_sessions:
        return _active_sessions[session_id]

    return ConversationManager()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@session_bp.route("/start", methods=["POST"])
def start_session():
    """
    Start a new conversation session.

    Request JSON:
        {
            "patient_id": "uuid" (optional),
            "patient_name": "Ravi" (optional),
            "language": "en" (optional),
            "context": "follow-up visit" (optional)
        }
    """
    data = request.get_json() or {}
    patient_id = data.get("patient_id")
    patient_name = data.get("patient_name")
    language = data.get("language", "en")
    context = data.get("context")

    try:
        manager = _get_manager()
        result = manager.start_session(
            patient_name=patient_name,
            language=language,
            context=context,
        )

        # Store in active sessions
        _active_sessions[manager.session_id] = manager

        # Persist to database
        _save_session_to_db(manager, patient_id)

        logger.info("Session started: %s", manager.session_id)
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Failed to start session")
        return jsonify({"error": str(e)}), 500


@session_bp.route("/<session_id>/message", methods=["POST"])
def send_message(session_id: str):
    """
    Send a message in an active session.

    Request JSON:
        { "message": "I have chest pain for 3 days" }
    """
    if session_id not in _active_sessions:
        return jsonify({"error": "Session not found. Start a new session."}), 404

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    try:
        manager = _active_sessions[session_id]
        result = manager.process_message(data["message"])

        # Update DB
        _update_session_in_db(manager)

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Message processing error")
        return jsonify({"error": str(e)}), 500


@session_bp.route("/<session_id>/summary", methods=["GET"])
def get_summary(session_id: str):
    """Generate a clinical summary for a session."""
    if session_id not in _active_sessions:
        # Try to load from DB
        return _get_summary_from_db(session_id)

    try:
        manager = _active_sessions[session_id]
        summary = manager.generate_summary()
        return jsonify(summary), 200

    except Exception as e:
        logger.exception("Summary generation error")
        return jsonify({"error": str(e)}), 500


@session_bp.route("/<session_id>/history", methods=["GET"])
def get_history(session_id: str):
    """Get conversation history for a session."""
    if session_id in _active_sessions:
        manager = _active_sessions[session_id]
        return jsonify({
            "session_id": session_id,
            "messages": manager.get_conversation_history(),
            "state": manager.symptom_logic.get_state(),
        }), 200

    # Load from DB
    return _get_history_from_db(session_id)


@session_bp.route("/<session_id>/end", methods=["POST"])
def end_session(session_id: str):
    """End a session, save summary to DB, and clean up."""
    if session_id not in _active_sessions:
        return jsonify({"error": "Session not found"}), 404

    try:
        manager = _active_sessions[session_id]
        result = manager.end_session()

        # Finalize in DB
        _finalize_session_in_db(manager, result)

        # Clean up
        del _active_sessions[session_id]
        logger.info("Session ended: %s", session_id)

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Session end error")
        return jsonify({"error": str(e)}), 500


@session_bp.route("/active", methods=["GET"])
def list_active_sessions():
    """List all active (in-memory) sessions."""
    sessions = []
    for sid, manager in _active_sessions.items():
        sessions.append({
            "session_id": sid,
            "patient_name": manager.patient_name,
            "started_at": manager.started_at.isoformat() if manager.started_at else None,
            "turns": manager.symptom_logic.conversation_turns,
            "symptoms": manager.symptom_logic.identified_symptoms,
        })
    return jsonify({"active_sessions": sessions, "count": len(sessions)}), 200


@session_bp.route("/all", methods=["GET"])
def list_all_sessions():
    """List all sessions from the database (paginated)."""
    from db import get_db
    from models import ConsultationSession

    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))
    status = request.args.get("status")

    db = get_db()
    try:
        query = db.query(ConsultationSession)
        if status:
            query = query.filter(ConsultationSession.status == status)

        total = query.count()
        sessions = (
            query
            .order_by(ConsultationSession.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

        return jsonify({
            "sessions": [s.to_dict() for s in sessions],
            "total": total,
            "limit": limit,
            "offset": offset,
        }), 200

    finally:
        db.close()


# ---------------------------------------------------------------------------
# DB persistence helpers
# ---------------------------------------------------------------------------

def _save_session_to_db(manager, patient_id=None):
    """Create a new session record in the database."""
    try:
        from db import get_db_session
        from models import ConsultationSession

        with get_db_session() as db:
            session_record = ConsultationSession(
                id=manager.session_id,
                patient_id=patient_id,
                patient_name=manager.patient_name,
                language=manager.language,
                status="active",
                model_used=manager.model,
            )
            db.add(session_record)

    except Exception as e:
        logger.error("Failed to save session to DB: %s", e)


def _update_session_in_db(manager):
    """Update session record with latest conversation state."""
    try:
        from db import get_db_session
        from models import ConsultationSession

        state = manager.symptom_logic.get_state()

        with get_db_session() as db:
            session_record = (
                db.query(ConsultationSession)
                .filter(ConsultationSession.id == manager.session_id)
                .first()
            )
            if session_record:
                session_record.conversation_history = json.dumps(
                    manager.get_conversation_history()
                )
                session_record.symptom_state = json.dumps(state)
                session_record.identified_symptoms = json.dumps(state.get("identified_symptoms", []))
                session_record.red_flags = json.dumps(state.get("red_flags", []))
                session_record.total_turns = state.get("conversation_turns", 0)
                session_record.total_messages = len(manager.get_conversation_history())
                session_record.assessment_complete = state.get("assessment_complete", False)

    except Exception as e:
        logger.error("Failed to update session in DB: %s", e)


def _finalize_session_in_db(manager, result):
    """Mark session as completed and save final summary."""
    try:
        from db import get_db_session
        from models import ConsultationSession

        with get_db_session() as db:
            session_record = (
                db.query(ConsultationSession)
                .filter(ConsultationSession.id == manager.session_id)
                .first()
            )
            if session_record:
                session_record.status = "completed"
                session_record.ended_at = datetime.now(timezone.utc)
                session_record.clinical_summary = result.get("clinical_summary")
                session_record.conversation_history = json.dumps(
                    manager.get_conversation_history()
                )

                # Extract chief complaint from symptoms
                symptoms = manager.symptom_logic.identified_symptoms
                if symptoms:
                    session_record.chief_complaint = symptoms[0]

    except Exception as e:
        logger.error("Failed to finalize session in DB: %s", e)


def _get_summary_from_db(session_id):
    """Load summary from DB for a completed session."""
    from db import get_db
    from models import ConsultationSession

    db = get_db()
    try:
        session = (
            db.query(ConsultationSession)
            .filter(ConsultationSession.id == session_id)
            .first()
        )
        if not session:
            return jsonify({"error": "Session not found"}), 404

        return jsonify({
            "session_id": session.id,
            "clinical_summary": session.clinical_summary,
            "identified_symptoms": json.loads(session.identified_symptoms or "[]"),
            "red_flags": json.loads(session.red_flags or "[]"),
            "conversation_turns": session.total_turns,
            "status": session.status,
        }), 200
    finally:
        db.close()


def _get_history_from_db(session_id):
    """Load conversation history from DB."""
    from db import get_db
    from models import ConsultationSession

    db = get_db()
    try:
        session = (
            db.query(ConsultationSession)
            .filter(ConsultationSession.id == session_id)
            .first()
        )
        if not session:
            return jsonify({"error": "Session not found"}), 404

        return jsonify({
            "session_id": session.id,
            "messages": json.loads(session.conversation_history or "[]"),
            "state": json.loads(session.symptom_state or "{}"),
        }), 200
    finally:
        db.close()
