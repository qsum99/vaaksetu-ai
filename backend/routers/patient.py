from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from backend.db.database import get_db, PatientDB
from backend.models.patient import Patient, PatientCreate

router = APIRouter(tags=["Patient"])

@router.post("/", response_model=Patient)
async def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    db_patient = PatientDB(**patient.model_dump())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@router.get("/list", response_model=List[Patient])
async def list_patients(db: Session = Depends(get_db)):
    return db.query(PatientDB).all()

@router.get("/search", response_model=List[Patient])
async def search_patients(q: str, db: Session = Depends(get_db)):
    # Basic search on name or village
    patients = db.query(PatientDB).filter(
        (PatientDB.name.ilike(f"%{q}%")) | (PatientDB.village.ilike(f"%{q}%"))
    ).all()
    return patients
