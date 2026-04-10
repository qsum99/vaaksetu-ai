"""
Patient Routes — CRUD operations for patient records + TTS endpoints.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

logger = logging.getLogger(__name__)

patient_bp = Blueprint("patient", __name__, url_prefix="/api")

# ---------------------------------------------------------------------------
# Lazy TTS singleton
# ---------------------------------------------------------------------------
_tts = None


def _get_tts():
    global _tts
    if _tts is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from ai_core.tts import SarvamTTS
        _tts = SarvamTTS()
    return _tts


# ===========================================================================
# Patient CRUD Endpoints
# ===========================================================================

@patient_bp.route("/patients", methods=["POST"])
def create_patient():
    """
    Create a new patient record.

    Request JSON:
        {
            "name": "Ravi Kumar",
            "age": 45,
            "gender": "male",
            "phone": "+91-9876543210",
            "preferred_language": "kn",
            ...
        }
    """
    from db import get_db_session
    from models import Patient

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    try:
        with get_db_session() as db:
            patient = Patient(
                name=data.get("name"),
                age=data.get("age"),
                gender=data.get("gender"),
                date_of_birth=_parse_date(data.get("date_of_birth")),
                phone=data.get("phone"),
                email=data.get("email"),
                weight_kg=data.get("weight_kg"),
                height_cm=data.get("height_cm"),
                blood_group=data.get("blood_group"),
                known_conditions=json.dumps(data.get("known_conditions", [])),
                known_allergies=json.dumps(data.get("known_allergies", [])),
                current_medications=json.dumps(data.get("current_medications", [])),
                preferred_language=data.get("preferred_language", "en"),
                address=data.get("address"),
                city=data.get("city"),
                state=data.get("state"),
                pincode=data.get("pincode"),
                notes=data.get("notes"),
            )
            db.add(patient)
            db.flush()  # Get ID before commit

            result = patient.to_dict()

        return jsonify(result), 201

    except Exception as e:
        logger.exception("Patient creation error")
        return jsonify({"error": str(e)}), 500


@patient_bp.route("/patients", methods=["GET"])
def list_patients():
    """
    List all patients with optional search.

    Query params:
        - search: Search by name or phone
        - limit: Max results (default 50)
        - offset: Pagination offset (default 0)
    """
    from db import get_db
    from models import Patient

    search = request.args.get("search", "").strip()
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))

    db = get_db()
    try:
        query = db.query(Patient)

        if search:
            query = query.filter(
                Patient.name.ilike(f"%{search}%")
                | Patient.phone.ilike(f"%{search}%")
            )

        total = query.count()
        patients = (
            query
            .order_by(Patient.updated_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

        return jsonify({
            "patients": [p.to_dict() for p in patients],
            "total": total,
            "limit": limit,
            "offset": offset,
        }), 200

    finally:
        db.close()


@patient_bp.route("/patients/<patient_id>", methods=["GET"])
def get_patient(patient_id):
    """Get a single patient by ID."""
    from db import get_db
    from models import Patient

    db = get_db()
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        return jsonify(patient.to_dict()), 200
    finally:
        db.close()


@patient_bp.route("/patients/<patient_id>", methods=["PUT"])
def update_patient(patient_id):
    """Update a patient record."""
    from db import get_db_session
    from models import Patient

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    try:
        with get_db_session() as db:
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                return jsonify({"error": "Patient not found"}), 404

            # Update fields
            updatable = [
                "name", "age", "gender", "phone", "email",
                "weight_kg", "height_cm", "blood_group",
                "preferred_language", "address", "city", "state",
                "pincode", "notes",
            ]
            for field in updatable:
                if field in data:
                    setattr(patient, field, data[field])

            # JSON fields
            for json_field in ["known_conditions", "known_allergies", "current_medications"]:
                if json_field in data:
                    setattr(patient, json_field, json.dumps(data[json_field]))

            if "date_of_birth" in data:
                patient.date_of_birth = _parse_date(data["date_of_birth"])

            result = patient.to_dict()

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Patient update error")
        return jsonify({"error": str(e)}), 500


@patient_bp.route("/patients/<patient_id>", methods=["DELETE"])
def delete_patient(patient_id):
    """Delete a patient and all related records."""
    from db import get_db_session
    from models import Patient

    try:
        with get_db_session() as db:
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            if not patient:
                return jsonify({"error": "Patient not found"}), 404

            db.delete(patient)

        return jsonify({"message": "Patient deleted", "id": patient_id}), 200

    except Exception as e:
        logger.exception("Patient deletion error")
        return jsonify({"error": str(e)}), 500


@patient_bp.route("/patients/<patient_id>/history", methods=["GET"])
def patient_history(patient_id):
    """Get all sessions and clinical records for a patient."""
    from db import get_db
    from models import Patient, ConsultationSession, ClinicalRecord

    db = get_db()
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        sessions = (
            db.query(ConsultationSession)
            .filter(ConsultationSession.patient_id == patient_id)
            .order_by(ConsultationSession.created_at.desc())
            .all()
        )

        records = (
            db.query(ClinicalRecord)
            .filter(ClinicalRecord.patient_id == patient_id)
            .order_by(ClinicalRecord.created_at.desc())
            .all()
        )

        return jsonify({
            "patient": patient.to_dict(),
            "sessions": [s.to_dict() for s in sessions],
            "clinical_records": [r.to_dict() for r in records],
            "total_sessions": len(sessions),
            "total_records": len(records),
        }), 200

    finally:
        db.close()


# ===========================================================================
# TTS Endpoints
# ===========================================================================

@patient_bp.route("/tts/synthesize", methods=["POST"])
def synthesize_speech():
    """
    Convert text to speech audio.

    Request JSON:
        { "text": "...", "language": "en", "format": "base64" | "binary" }
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"]
    language = data.get("language", "en")
    voice = data.get("voice")
    output_format = data.get("format", "base64")

    if not text.strip():
        return jsonify({"error": "Empty text provided"}), 400

    try:
        tts = _get_tts()

        if output_format == "binary":
            audio_bytes = tts.synthesize(text, language, voice)
            return Response(
                audio_bytes,
                mimetype="audio/wav",
                headers={"Content-Disposition": "inline; filename=speech.wav"},
            )
        else:
            result = tts.synthesize_base64(text, language, voice)
            return jsonify(result), 200

    except Exception as e:
        logger.exception("TTS synthesis error")
        return jsonify({"error": str(e)}), 500


@patient_bp.route("/tts/voices", methods=["GET"])
def list_voices():
    """Return available TTS voices."""
    tts = _get_tts()
    return jsonify(tts.get_available_voices()), 200


# ===========================================================================
# Helpers
# ===========================================================================

def _parse_date(date_str):
    """Parse ISO date string to datetime object."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None
