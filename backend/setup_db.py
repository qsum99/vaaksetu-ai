"""
Database setup script — Initialize tables and seed sample data.

Run this once to set up the database:
    python setup_db.py

Options:
    python setup_db.py --seed     # Create tables + seed sample data
    python setup_db.py --reset    # Drop all tables, recreate, and seed
"""

import json
import sys
from datetime import datetime, timezone


def setup():
    """Create all database tables."""
    from db import init_db
    print("Creating database tables...")
    init_db()
    print("[OK] Database tables created successfully.")


def seed():
    """Seed the database with sample data for development."""
    from db import get_db_session
    from models import Patient, ConsultationSession, ClinicalRecord

    print("Seeding sample data...")

    with get_db_session() as db:
        # Sample patients
        patients = [
            Patient(
                name="Ravi Kumar",
                age=45,
                gender="male",
                phone="+91-9876543210",
                blood_group="O+",
                known_conditions=json.dumps(["diabetes", "hypertension"]),
                known_allergies=json.dumps(["penicillin"]),
                current_medications=json.dumps(["metformin 500mg", "amlodipine 5mg"]),
                preferred_language="kn",
                city="Bengaluru",
                state="Karnataka",
                pincode="560001",
            ),
            Patient(
                name="Priya Sharma",
                age=32,
                gender="female",
                phone="+91-9876543211",
                blood_group="B+",
                known_conditions=json.dumps(["asthma"]),
                known_allergies=json.dumps([]),
                current_medications=json.dumps(["salbutamol inhaler"]),
                preferred_language="hi",
                city="Mumbai",
                state="Maharashtra",
                pincode="400001",
            ),
            Patient(
                name="Anand Reddy",
                age=58,
                gender="male",
                phone="+91-9876543212",
                blood_group="A+",
                known_conditions=json.dumps(["arthritis"]),
                known_allergies=json.dumps(["sulfa drugs"]),
                current_medications=json.dumps([]),
                preferred_language="te",
                city="Hyderabad",
                state="Telangana",
                pincode="500001",
            ),
        ]

        for p in patients:
            db.add(p)
        db.flush()

        # Sample session for Ravi
        session = ConsultationSession(
            patient_id=patients[0].id,
            patient_name="Ravi Kumar",
            language="kn",
            status="completed",
            ended_at=datetime.now(timezone.utc),
            chief_complaint="chest pain",
            clinical_summary="Patient presents with chest pain for 3 days...",
            identified_symptoms=json.dumps(["chest pain", "breathlessness"]),
            red_flags=json.dumps([]),
            conversation_history=json.dumps([
                {"role": "assistant", "content": "Hello Ravi! How are you today?"},
                {"role": "user", "content": "I have chest pain for 3 days"},
                {"role": "assistant", "content": "I understand. Can you describe the pain?"},
                {"role": "user", "content": "It is a dull ache in the center"},
            ]),
            total_turns=4,
            total_messages=4,
            assessment_complete=True,
            model_used="llama-3.3-70b-versatile",
        )
        db.add(session)
        db.flush()

        # Sample clinical record
        record = ClinicalRecord(
            patient_id=patients[0].id,
            session_id=session.id,
            chief_complaint="chest pain for 3 days",
            language_detected="en",
            confidence_score=0.85,
            completeness_score=0.6,
            extraction_mode="rules",
            record_json=json.dumps({
                "chief_complaint": "chest pain for 3 days",
                "symptoms": [
                    {"name": "chest pain", "duration": "3 days", "character": "dull ache"},
                    {"name": "breathlessness", "severity": "mild"},
                ],
                "vitals": {"blood_pressure_systolic": 140, "blood_pressure_diastolic": 90},
            }),
            symptoms_json=json.dumps(["chest pain", "breathlessness"]),
            diagnosis_json=json.dumps([]),
            bp_systolic=140,
            bp_diastolic=90,
            has_red_flags=False,
        )
        db.add(record)

    print("[OK] Sample data seeded: 3 patients, 1 session, 1 clinical record.")


def reset():
    """Drop all tables and recreate with seed data."""
    from db import drop_db, init_db
    print("[WARN] Dropping all tables...")
    drop_db()
    print("Creating fresh tables...")
    init_db()
    print("Seeding data...")
    seed()
    print("[OK] Database reset complete.")


if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset()
    elif "--seed" in sys.argv:
        setup()
        seed()
    else:
        setup()
