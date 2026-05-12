"""Report routes — save and retrieve clinical records."""
from datetime import date
from flask import Blueprint, request, jsonify
from models.database import get_db
from models.patient import PatientDB
from models.clinical_record import ClinicalRecordDB

report_bp = Blueprint("report", __name__, url_prefix="/api/report")


@report_bp.route("/save", methods=["POST"])
def save_report():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    db = get_db()
    try:
        # Find or create patient
        patient_name = data.get("patient_name", "Unknown")
        patient = db.query(PatientDB).filter(PatientDB.name == patient_name).first()

        if not patient:
            patient = PatientDB(
                name=patient_name,
                age=data.get("age", 0),
                gender=data.get("gender", "other"),
                village=data.get("village", "Unknown"),
            )
            db.add(patient)
            db.commit()
            db.refresh(patient)

        # Parse followup_date safely
        followup_date = None
        if data.get("followup_date"):
            try:
                followup_date = date.fromisoformat(data["followup_date"])
            except (ValueError, TypeError):
                pass

        # Parse created_at safely
        created_at = date.today()
        if data.get("created_at"):
            try:
                created_at = date.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                pass

        db_record = ClinicalRecordDB(
            patient_id=patient.id,
            patient_name=patient_name,
            age=data.get("age", 0),
            gender=data.get("gender", "other"),
            village=data.get("village", "Unknown"),
            chief_complaint=data.get("chief_complaint", ""),
            symptoms=data.get("symptoms", []),
            duration=data.get("duration", "Not specified"),
            vitals=data.get("vitals", {}),
            medical_history=data.get("medical_history", ""),
            diagnosis_notes=data.get("diagnosis_notes", ""),
            followup_date=followup_date,
            raw_transcript=data.get("raw_transcript", ""),
            language_mix=data.get("language_mix", ["en"]),
            created_at=created_at,
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return jsonify({"status": "success", "record_id": db_record.id}), 201

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500


@report_bp.route("/list", methods=["GET"])
def list_reports():
    db = get_db()
    records = (
        db.query(ClinicalRecordDB)
        .order_by(ClinicalRecordDB.created_at.desc())
        .all()
    )
    return jsonify([r.to_dict() for r in records])


@report_bp.route("/<int:record_id>", methods=["GET"])
def get_report(record_id):
    db = get_db()
    record = db.query(ClinicalRecordDB).filter(ClinicalRecordDB.id == record_id).first()
    if not record:
        return jsonify({"error": "Report not found"}), 404
    return jsonify(record.to_dict())
