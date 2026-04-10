"""
Report Routes — Clinical data extraction with DB persistence.
"""

import json
import logging
import sys
from pathlib import Path

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

report_bp = Blueprint("report", __name__, url_prefix="/api/report")

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------
_extractor = None


def _get_extractor():
    global _extractor
    if _extractor is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from ai_core.extraction import ClinicalExtractor
        _extractor = ClinicalExtractor()
    return _extractor


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@report_bp.route("/extract", methods=["POST"])
def extract_clinical_data():
    """
    Extract structured clinical data from a transcript and save to DB.

    Request JSON:
        {
            "transcript": "Doctor: ...\\nPatient: ...",
            "language": "en",
            "mode": "llm" | "rules",
            "patient_id": "uuid" (optional),
            "session_id": "uuid" (optional)
        }
    """
    data = request.get_json()
    if not data or "transcript" not in data:
        return jsonify({"error": "Missing 'transcript' in request body"}), 400

    transcript = data["transcript"]
    language = data.get("language", "en")
    mode = data.get("mode", "llm")
    patient_id = data.get("patient_id")
    session_id = data.get("session_id")

    try:
        extractor = _get_extractor()

        if mode == "rules":
            record = extractor.extract_rules_based(transcript, language)
        else:
            record = extractor.extract(transcript, language)

        # Save to database
        db_record_id = _save_record_to_db(record, patient_id, session_id, mode)

        response = record.model_dump(exclude_none=False)
        response["completeness"] = round(record.completeness_score(), 2)
        response["db_record_id"] = db_record_id

        return jsonify(response), 200

    except Exception as e:
        logger.exception("Extraction error")
        return jsonify({"error": str(e)}), 500


@report_bp.route("/extract-incremental", methods=["POST"])
def extract_incremental():
    """Incrementally extract and merge new data."""
    data = request.get_json()
    if not data or "transcript" not in data:
        return jsonify({"error": "Missing 'transcript' in request body"}), 400

    transcript = data["transcript"]
    language = data.get("language", "en")
    existing_data = data.get("existing_record", {})

    try:
        from ai_core.extraction import ClinicalRecord

        existing_record = ClinicalRecord.model_validate(existing_data)
        extractor = _get_extractor()
        merged = extractor.extract_incremental(transcript, existing_record, language)

        response = merged.model_dump(exclude_none=False)
        response["completeness"] = round(merged.completeness_score(), 2)

        return jsonify(response), 200

    except Exception as e:
        logger.exception("Incremental extraction error")
        return jsonify({"error": str(e)}), 500


@report_bp.route("/records", methods=["GET"])
def list_records():
    """
    List clinical records with optional filters.

    Query params:
        - patient_id: Filter by patient
        - session_id: Filter by session
        - has_red_flags: true/false
        - limit, offset: Pagination
    """
    from db import get_db
    from models.clinical_record import ClinicalRecord as ClinicalRecordModel

    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))
    patient_id = request.args.get("patient_id")
    session_id = request.args.get("session_id")
    has_red_flags = request.args.get("has_red_flags")

    db = get_db()
    try:
        query = db.query(ClinicalRecordModel)

        if patient_id:
            query = query.filter(ClinicalRecordModel.patient_id == patient_id)
        if session_id:
            query = query.filter(ClinicalRecordModel.session_id == session_id)
        if has_red_flags is not None:
            query = query.filter(ClinicalRecordModel.has_red_flags == (has_red_flags == "true"))

        total = query.count()
        records = (
            query
            .order_by(ClinicalRecordModel.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

        return jsonify({
            "records": [r.to_dict() for r in records],
            "total": total,
            "limit": limit,
            "offset": offset,
        }), 200

    finally:
        db.close()


@report_bp.route("/records/<record_id>", methods=["GET"])
def get_record(record_id):
    """Get a single clinical record by ID."""
    from db import get_db
    from models.clinical_record import ClinicalRecord as ClinicalRecordModel

    db = get_db()
    try:
        record = db.query(ClinicalRecordModel).filter(ClinicalRecordModel.id == record_id).first()
        if not record:
            return jsonify({"error": "Record not found"}), 404

        return jsonify(record.to_dict()), 200
    finally:
        db.close()


@report_bp.route("/records/<record_id>/review", methods=["POST"])
def review_record(record_id):
    """
    Mark a clinical record as reviewed by a doctor.

    Request JSON:
        { "doctor_notes": "Confirmed diagnosis..." }
    """
    from db import get_db_session
    from models.clinical_record import ClinicalRecord as ClinicalRecordModel
    from datetime import datetime, timezone

    data = request.get_json() or {}

    try:
        with get_db_session() as db:
            record = db.query(ClinicalRecordModel).filter(ClinicalRecordModel.id == record_id).first()
            if not record:
                return jsonify({"error": "Record not found"}), 404

            record.reviewed_by_doctor = True
            record.doctor_notes = data.get("doctor_notes")
            record.reviewed_at = datetime.now(timezone.utc)

            result = record.to_dict()

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Review error")
        return jsonify({"error": str(e)}), 500


@report_bp.route("/schema", methods=["GET"])
def get_schema():
    """Return the clinical record JSON schema."""
    schema_path = Path(__file__).resolve().parents[2] / "ai_core" / "extraction" / "clinical_schema.json"
    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
        return jsonify(schema), 200
    except FileNotFoundError:
        from ai_core.extraction import ClinicalRecord
        return jsonify(ClinicalRecord.model_json_schema()), 200


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

def _save_record_to_db(record, patient_id=None, session_id=None, mode="llm"):
    """Save an extracted ClinicalRecord to the database."""
    try:
        from db import get_db_session
        from models.clinical_record import ClinicalRecord as ClinicalRecordModel

        record_dict = record.model_dump(exclude_none=False)
        vitals = record_dict.get("vitals", {})

        with get_db_session() as db:
            db_record = ClinicalRecordModel(
                patient_id=patient_id,
                session_id=session_id,
                chief_complaint=record_dict.get("chief_complaint"),
                language_detected=record_dict.get("language_detected"),
                confidence_score=record_dict.get("confidence_score"),
                completeness_score=round(record.completeness_score(), 2),
                extraction_mode=mode,
                record_json=json.dumps(record_dict),
                symptoms_json=json.dumps([s["name"] for s in record_dict.get("symptoms", [])]),
                diagnosis_json=json.dumps(record_dict.get("provisional_diagnosis", [])),
                medications_json=json.dumps([m["name"] for m in record_dict.get("prescribed_medications", [])]),
                tests_json=json.dumps(record_dict.get("recommended_tests", [])),
                bp_systolic=vitals.get("blood_pressure_systolic"),
                bp_diastolic=vitals.get("blood_pressure_diastolic"),
                heart_rate=vitals.get("heart_rate_bpm"),
                temperature=vitals.get("temperature_f"),
                spo2=vitals.get("spo2_percent"),
                has_red_flags=False,  # TODO: detect from record
            )
            db.add(db_record)
            db.flush()
            return db_record.id

    except Exception as e:
        logger.error("Failed to save clinical record to DB: %s", e)
        return None
