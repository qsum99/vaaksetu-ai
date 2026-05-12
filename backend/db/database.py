import json
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, Date, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import date
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vaaksetu.db")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PatientDB(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    village = Column(String, index=True)
    created_at = Column(Date, default=date.today)

    reports = relationship("ClinicalRecordDB", back_populates="patient")

class ClinicalRecordDB(Base):
    __tablename__ = "clinical_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    patient_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    village = Column(String)
    chief_complaint = Column(Text)
    symptoms = Column(JSON)
    duration = Column(String)
    vitals = Column(JSON, nullable=True)
    medical_history = Column(Text, nullable=True)
    diagnosis_notes = Column(Text, nullable=True)
    followup_date = Column(Date, nullable=True)
    raw_transcript = Column(Text)
    language_mix = Column(JSON)
    created_at = Column(Date, default=date.today)

    patient = relationship("PatientDB", back_populates="reports")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_sample_data():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "patients.json")
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
    except FileNotFoundError:
        return

    db = SessionLocal()
    try:
        if db.query(PatientDB).count() > 0:
            return

        for record in sample_data:
            patient = PatientDB(
                name=record.get("patient_name", "Unknown"),
                age=record.get("age", 0),
                gender=record.get("gender", "other"),
                village=record.get("village", "Unknown"),
                created_at=date.fromisoformat(record.get("created_at")) if record.get("created_at") else date.today()
            )
            db.add(patient)
            db.commit()
            db.refresh(patient)

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
                followup_date=date.fromisoformat(record.get("followup_date")) if record.get("followup_date") else None,
                raw_transcript=record.get("raw_transcript", ""),
                language_mix=record.get("language_mix", ["en"]),
                created_at=date.fromisoformat(record.get("created_at")) if record.get("created_at") else date.today()
            )
            db.add(db_record)
            db.commit()
    finally:
        db.close()
