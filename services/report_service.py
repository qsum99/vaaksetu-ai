"""Report service — business logic for clinical record operations."""
from datetime import date
from models.database import get_db
from models.patient import PatientDB
from models.clinical_record import ClinicalRecordDB


def save_clinical_record(data: dict) -> dict:
    """Create a clinical record + auto-create patient if needed."""
    db = get_db()

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

    followup_date = None
    if data.get("followup_date"):
        try:
            followup_date = date.fromisoformat(data["followup_date"])
        except (ValueError, TypeError):
            pass

    created_at = date.today()
    if data.get("created_at"):
        try:
            created_at = date.fromisoformat(data["created_at"])
        except (ValueError, TypeError):
            pass

    record = ClinicalRecordDB(
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
    db.add(record)
    db.commit()
    db.refresh(record)

    return {"status": "success", "record_id": record.id}


def get_record_by_id(record_id: int):
    db = get_db()
    record = db.query(ClinicalRecordDB).filter(ClinicalRecordDB.id == record_id).first()
    return record.to_dict() if record else None


def list_all_records():
    db = get_db()
    records = db.query(ClinicalRecordDB).order_by(ClinicalRecordDB.created_at.desc()).all()
    return [r.to_dict() for r in records]
