"""Patient routes — CRUD operations for patients."""
from flask import Blueprint, request, jsonify
from models.database import get_db, close_db
from models.patient import PatientDB

patient_bp = Blueprint("patient", __name__, url_prefix="/api/patient")


@patient_bp.route("/", methods=["POST"])
def create_patient():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Patient name is required"}), 400

    db = get_db()
    try:
        patient = PatientDB(
            name=data["name"],
            age=data.get("age", 0),
            gender=data.get("gender", "other"),
            village=data.get("village", "Unknown"),
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)
        return jsonify(patient.to_dict()), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500


@patient_bp.route("/list", methods=["GET"])
def list_patients():
    db = get_db()
    patients = db.query(PatientDB).order_by(PatientDB.created_at.desc()).all()
    return jsonify([p.to_dict() for p in patients])


@patient_bp.route("/search", methods=["GET"])
def search_patients():
    q = request.args.get("q", "")
    if not q:
        return list_patients()

    db = get_db()
    patients = (
        db.query(PatientDB)
        .filter(PatientDB.name.ilike(f"%{q}%") | PatientDB.village.ilike(f"%{q}%"))
        .all()
    )
    return jsonify([p.to_dict() for p in patients])
