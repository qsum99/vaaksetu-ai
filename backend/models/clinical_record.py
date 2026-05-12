from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date

class ClinicalRecord(BaseModel):
    patient_name: str
    age: int
    gender: str
    village: str
    chief_complaint: str
    symptoms: List[str]
    duration: str
    vitals: Optional[dict] = Field(default_factory=dict)
    medical_history: Optional[str] = None
    diagnosis_notes: Optional[str] = None
    followup_date: Optional[date] = None
    raw_transcript: str
    language_mix: List[str]
    created_at: date
