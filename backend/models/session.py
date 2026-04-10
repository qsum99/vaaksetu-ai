"""
ConsultationSession Model — Tracks conversation sessions.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, ForeignKey,
    Enum as SAEnum, Boolean
)
from sqlalchemy.orm import relationship

from db.database import Base


class ConsultationSession(Base):
    """
    Consultation session table — one per doctor-patient conversation.

    Stores the full conversation history, symptom state, and metadata
    for each session. Links to a patient and may produce one or more
    clinical records.
    """
    __tablename__ = "consultation_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    ended_at = Column(DateTime, nullable=True)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Session metadata
    status = Column(
        SAEnum("active", "completed", "abandoned", name="session_status_enum"),
        default="active",
        nullable=False,
    )
    language = Column(String(5), default="en", nullable=False)
    patient_name = Column(String(255), nullable=True)

    # Conversation data (stored as JSON strings)
    conversation_history = Column(Text, nullable=True)  # JSON: [{role, content}, ...]
    symptom_state = Column(Text, nullable=True)          # JSON: SymptomLogic state
    identified_symptoms = Column(Text, nullable=True)    # JSON: list of symptom strings
    red_flags = Column(Text, nullable=True)              # JSON: list of red flag dicts

    # Summary
    clinical_summary = Column(Text, nullable=True)
    chief_complaint = Column(String(500), nullable=True)

    # Stats
    total_turns = Column(Integer, default=0, nullable=False)
    total_messages = Column(Integer, default=0, nullable=False)
    assessment_complete = Column(Boolean, default=False, nullable=False)

    # AI metadata
    model_used = Column(String(100), nullable=True)
    total_tokens_used = Column(Integer, default=0, nullable=True)

    # Relationships
    patient = relationship("Patient", back_populates="sessions")
    clinical_records = relationship("ClinicalRecord", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session id={self.id} patient={self.patient_name} status={self.status}>"

    def to_dict(self):
        """Serialize to dictionary."""
        import json
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "status": self.status,
            "language": self.language,
            "chief_complaint": self.chief_complaint,
            "clinical_summary": self.clinical_summary,
            "identified_symptoms": json.loads(self.identified_symptoms) if self.identified_symptoms else [],
            "red_flags": json.loads(self.red_flags) if self.red_flags else [],
            "conversation_history": json.loads(self.conversation_history) if self.conversation_history else [],
            "total_turns": self.total_turns,
            "total_messages": self.total_messages,
            "assessment_complete": self.assessment_complete,
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }
