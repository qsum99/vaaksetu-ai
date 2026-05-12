from pydantic import BaseModel
from typing import Optional
from datetime import date

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    village: str

class Patient(PatientCreate):
    id: int
    created_at: date

    class Config:
        from_attributes = True
