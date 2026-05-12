from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from backend.db.database import get_db, ClinicalRecordDB, PatientDB
from backend.models.clinical_record import ClinicalRecord

router = APIRouter(tags=["Report"])

class ClinicalRecordResponse(ClinicalRecord):
    id: int
    patient_id: int

@router.post("/save")
async def save_report(record: ClinicalRecord, db: Session = Depends(get_db)):
    # First, try to find the patient or create a new one
    patient = db.query(PatientDB).filter(PatientDB.name == record.patient_name).first()
    if not patient:
        patient = PatientDB(
            name=record.patient_name,
            age=record.age,
            gender=record.gender,
            village=record.village
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)
    
    # Save the clinical record
    db_record = ClinicalRecordDB(
        patient_id=patient.id,
        **record.model_dump()
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {"status": "success", "record_id": db_record.id}

@router.get("/{record_id}")
async def get_report(record_id: int, db: Session = Depends(get_db)):
    record = db.query(ClinicalRecordDB).filter(ClinicalRecordDB.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Report not found")
    return record
