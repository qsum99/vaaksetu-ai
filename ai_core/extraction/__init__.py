"""
Vaaksetu Extraction Package — Structured clinical data extraction.

Extracts structured medical records from doctor-patient conversation
transcripts using LLM-powered and rule-based approaches.

Quick start::

    from ai_core.extraction import ClinicalExtractor, ClinicalRecord

    extractor = ClinicalExtractor(api_key="sk-...")
    record = extractor.extract(
        "Patient: I have severe chest pain for 3 days. "
        "Doctor: Any radiation to the left arm?"
    )
    print(record.symptoms[0].name)       # "chest pain"
    print(record.symptoms[0].severity)   # "severe"
    print(record.model_dump_json(indent=2))
"""

from .models import (
    ClinicalRecord,
    PatientInfo,
    Symptom,
    Vitals,
    MedicalHistory,
    Medication,
    PrescribedMedication,
    Lifestyle,
    FollowUp,
    Gender,
    Severity,
)
from .extractor import ClinicalExtractor, ExtractionError

__all__ = [
    # Main classes
    "ClinicalExtractor",
    "ClinicalRecord",
    "ExtractionError",
    # Sub-models
    "PatientInfo",
    "Symptom",
    "Vitals",
    "MedicalHistory",
    "Medication",
    "PrescribedMedication",
    "Lifestyle",
    "FollowUp",
    # Enums
    "Gender",
    "Severity",
]
