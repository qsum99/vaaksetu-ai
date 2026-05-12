"""ClinicalRecordDB — SQLAlchemy model for clinical_records table."""
from datetime import date
from sqlalchemy import Column, Integer, String, Text, JSON, Date, ForeignKey
from sqlalchemy.orm import relationship
from models.database import Base


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

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "age": self.age,
            "gender": self.gender,
            "village": self.village,
            "chief_complaint": self.chief_complaint,
            "symptoms": self.symptoms or [],
            "duration": self.duration,
            "vitals": self.vitals or {},
            "medical_history": self.medical_history,
            "diagnosis_notes": self.diagnosis_notes,
            "followup_date": str(self.followup_date) if self.followup_date else None,
            "raw_transcript": self.raw_transcript,
            "language_mix": self.language_mix or [],
            "created_at": str(self.created_at) if self.created_at else None,
        }
