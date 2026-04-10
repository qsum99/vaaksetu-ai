"""
Patient Model — SQLAlchemy ORM model for patient records.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Enum as SAEnum
)
from sqlalchemy.orm import relationship

from db.database import Base


class Patient(Base):
    """
    Patient table — stores demographic and contact information.

    Each patient can have multiple consultation sessions and
    clinical records over time (longitudinal tracking).
    """
    __tablename__ = "patients"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Demographics
    name = Column(String(255), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(SAEnum("male", "female", "other", name="gender_enum"), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    phone = Column(String(20), nullable=True, index=True)
    email = Column(String(255), nullable=True)

    # Physical
    weight_kg = Column(Float, nullable=True)
    height_cm = Column(Float, nullable=True)
    blood_group = Column(String(10), nullable=True)

    # Medical background
    known_conditions = Column(Text, nullable=True)  # JSON array as string
    known_allergies = Column(Text, nullable=True)    # JSON array as string
    current_medications = Column(Text, nullable=True) # JSON array as string

    # Address
    address = Column(Text, nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    pincode = Column(String(10), nullable=True)

    # Preferences
    preferred_language = Column(String(5), default="en", nullable=False)

    # Notes
    notes = Column(Text, nullable=True)

    # Relationships
    sessions = relationship("ConsultationSession", back_populates="patient", cascade="all, delete-orphan")
    clinical_records = relationship("ClinicalRecord", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient id={self.id} name={self.name} age={self.age}>"

    def to_dict(self):
        """Serialize to dictionary."""
        import json
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "date_of_birth": self.date_of_birth.isoformat() if self.date_of_birth else None,
            "phone": self.phone,
            "email": self.email,
            "weight_kg": self.weight_kg,
            "height_cm": self.height_cm,
            "blood_group": self.blood_group,
            "known_conditions": json.loads(self.known_conditions) if self.known_conditions else [],
            "known_allergies": json.loads(self.known_allergies) if self.known_allergies else [],
            "current_medications": json.loads(self.current_medications) if self.current_medications else [],
            "preferred_language": self.preferred_language,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "pincode": self.pincode,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
