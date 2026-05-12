"""PatientDB — SQLAlchemy model for the patients table."""
from datetime import date
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.orm import relationship
from models.database import Base


class PatientDB(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    age = Column(Integer, default=0)
    gender = Column(String, default="other")
    village = Column(String, index=True, default="Unknown")
    created_at = Column(Date, default=date.today)

    reports = relationship("ClinicalRecordDB", back_populates="patient")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "village": self.village,
            "created_at": str(self.created_at) if self.created_at else None,
        }
