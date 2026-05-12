"""
VaakSetu — Database engine, session factory, and seed logic.
Uses scoped_session so Flask routes can just call db_session() directly.
"""
import json
import os
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

from config import Config

engine = create_engine(
    Config.DATABASE_URL,
    connect_args=Config.SQLALCHEMY_CONNECT_ARGS,
)

session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(session_factory)

Base = declarative_base()


def init_db():
    """Create all tables."""
    # Import models so they register with Base.metadata
    import models.patient  # noqa: F401
    import models.clinical_record  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """Return a DB session (non-generator, Flask-friendly)."""
    return db_session()


def close_db(exception=None):
    """Teardown helper — call from Flask's teardown_appcontext."""
    db_session.remove()


def seed_sample_data():
    """Populate the DB with sample patients if empty."""
    from models.patient import PatientDB
    from models.clinical_record import ClinicalRecordDB

    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "sample_data", "patients.json"
    )
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
    except FileNotFoundError:
        print(f"[seed] sample data not found at {sample_path}")
        return

    session = db_session()
    try:
        if session.query(PatientDB).count() > 0:
            return

        for record in sample_data:
            patient = PatientDB(
                name=record.get("patient_name", "Unknown"),
                age=record.get("age", 0),
                gender=record.get("gender", "other"),
                village=record.get("village", "Unknown"),
                created_at=(
                    date.fromisoformat(record["created_at"])
                    if record.get("created_at")
                    else date.today()
                ),
            )
            session.add(patient)
            session.commit()
            session.refresh(patient)

            db_record = ClinicalRecordDB(
                patient_id=patient.id,
                patient_name=record.get("patient_name", "Unknown"),
                age=record.get("age", 0),
                gender=record.get("gender", "other"),
                village=record.get("village", "Unknown"),
                chief_complaint=record.get("chief_complaint", ""),
                symptoms=record.get("symptoms", []),
                duration=record.get("duration", ""),
                vitals=record.get("vitals", {}),
                medical_history=record.get("medical_history", ""),
                diagnosis_notes=record.get("diagnosis_notes", ""),
                followup_date=(
                    date.fromisoformat(record["followup_date"])
                    if record.get("followup_date")
                    else None
                ),
                raw_transcript=record.get("raw_transcript", ""),
                language_mix=record.get("language_mix", ["en"]),
                created_at=(
                    date.fromisoformat(record["created_at"])
                    if record.get("created_at")
                    else date.today()
                ),
            )
            session.add(db_record)
            session.commit()

        print("[seed] Sample data loaded successfully.")
    finally:
        db_session.remove()
