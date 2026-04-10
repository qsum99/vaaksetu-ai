"""
ClinicalRecord Model — Structured medical data from extraction.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, ForeignKey, Boolean
)
from sqlalchemy.orm import relationship

from db.database import Base


class ClinicalRecord(Base):
    """
    Clinical record table — stores structured extracted medical data.

    Each record corresponds to one consultation session's extracted
    data. Stores the full Pydantic ClinicalRecord as JSON plus
    key fields denormalized for querying.
    """
    __tablename__ = "clinical_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=True, index=True)
    session_id = Column(String(36), ForeignKey("consultation_sessions.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Denormalized key fields (for querying / dashboards)
    chief_complaint = Column(String(500), nullable=True)
    language_detected = Column(String(5), nullable=True)
    confidence_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)

    # Extraction mode
    extraction_mode = Column(String(20), default="llm", nullable=False)  # "llm" or "rules"

    # Full structured record (JSON blob)
    record_json = Column(Text, nullable=False)  # Full ClinicalRecord as JSON

    # Denormalized symptoms (for search)
    symptoms_json = Column(Text, nullable=True)      # JSON array of symptom names
    diagnosis_json = Column(Text, nullable=True)      # JSON array of diagnoses
    medications_json = Column(Text, nullable=True)    # JSON array of medication names
    tests_json = Column(Text, nullable=True)          # JSON array of recommended tests

    # Vitals snapshot
    bp_systolic = Column(Integer, nullable=True)
    bp_diastolic = Column(Integer, nullable=True)
    heart_rate = Column(Integer, nullable=True)
    temperature = Column(Float, nullable=True)
    spo2 = Column(Float, nullable=True)

    # Red flags
    has_red_flags = Column(Boolean, default=False, nullable=False)
    red_flags_json = Column(Text, nullable=True)

    # Doctor review
    reviewed_by_doctor = Column(Boolean, default=False, nullable=False)
    doctor_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    # Relationships
    patient = relationship("Patient", back_populates="clinical_records")
    session = relationship("ConsultationSession", back_populates="clinical_records")

    def __repr__(self):
        return f"<ClinicalRecord id={self.id} complaint={self.chief_complaint}>"

    def to_dict(self):
        """Serialize to dictionary."""
        import json
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "chief_complaint": self.chief_complaint,
            "language_detected": self.language_detected,
            "confidence_score": self.confidence_score,
            "completeness_score": self.completeness_score,
            "extraction_mode": self.extraction_mode,
            "record": json.loads(self.record_json) if self.record_json else {},
            "symptoms": json.loads(self.symptoms_json) if self.symptoms_json else [],
            "diagnosis": json.loads(self.diagnosis_json) if self.diagnosis_json else [],
            "medications": json.loads(self.medications_json) if self.medications_json else [],
            "tests": json.loads(self.tests_json) if self.tests_json else [],
            "vitals": {
                "bp_systolic": self.bp_systolic,
                "bp_diastolic": self.bp_diastolic,
                "heart_rate": self.heart_rate,
                "temperature": self.temperature,
                "spo2": self.spo2,
            },
            "has_red_flags": self.has_red_flags,
            "red_flags": json.loads(self.red_flags_json) if self.red_flags_json else [],
            "reviewed_by_doctor": self.reviewed_by_doctor,
            "doctor_notes": self.doctor_notes,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
