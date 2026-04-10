"""
Pydantic Clinical Models — Typed schemas for structured medical data.

These models mirror the clinical_schema.json and provide runtime validation,
serialization, and IDE autocompletion for extracted medical records.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class PatientInfo(BaseModel):
    """Basic patient demographics mentioned in conversation."""
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[Gender] = None
    weight_kg: Optional[float] = Field(None, ge=0)
    height_cm: Optional[float] = Field(None, ge=0)


class Symptom(BaseModel):
    """
    A single symptom reported by the patient.

    Follows the OPQRST mnemonic used in clinical assessment:
    Onset, Provocation/Palliation, Quality, Radiation, Severity, Time.
    """
    name: str = Field(..., description="Symptom name (e.g. 'chest pain')")
    body_region: Optional[str] = Field(None, description="Affected body area")
    severity: Optional[Severity] = None
    duration: Optional[str] = Field(None, description="e.g. '3 days', '2 weeks'")
    frequency: Optional[str] = Field(None, description="e.g. 'constant', 'intermittent'")
    onset: Optional[str] = Field(None, description="When / what triggered it")
    character: Optional[str] = Field(None, description="e.g. 'sharp', 'dull', 'burning'")
    radiation: Optional[str] = Field(None, description="Where it spreads to")
    aggravating_factors: List[str] = Field(default_factory=list)
    relieving_factors: List[str] = Field(default_factory=list)
    associated_symptoms: List[str] = Field(default_factory=list)


class Vitals(BaseModel):
    """Vital signs mentioned or measured during the encounter."""
    blood_pressure_systolic: Optional[int] = Field(None, ge=0, le=300)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=0, le=200)
    heart_rate_bpm: Optional[int] = Field(None, ge=0, le=300)
    temperature_f: Optional[float] = Field(None, ge=85, le=115)
    respiratory_rate: Optional[int] = Field(None, ge=0, le=60)
    spo2_percent: Optional[float] = Field(None, ge=0, le=100)
    blood_sugar_mg_dl: Optional[float] = Field(None, ge=0)


class MedicalHistory(BaseModel):
    """Past medical, surgical, and family history."""
    past_conditions: List[str] = Field(default_factory=list)
    surgeries: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)


class Medication(BaseModel):
    """A medication the patient is currently taking."""
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = Field(None, description="oral, IV, topical, etc.")


class PrescribedMedication(BaseModel):
    """A new medication prescribed during this visit."""
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instructions: Optional[str] = None


class Lifestyle(BaseModel):
    """Lifestyle and social history factors."""
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    diet: Optional[str] = None
    exercise: Optional[str] = None
    sleep: Optional[str] = None
    occupation: Optional[str] = None


class FollowUp(BaseModel):
    """Follow-up instructions and red flags."""
    when: Optional[str] = Field(None, description="e.g. '1 week', '3 days'")
    instructions: Optional[str] = None
    red_flags: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class ClinicalRecord(BaseModel):
    """
    Complete structured clinical record extracted from a
    doctor-patient conversation.

    This is the primary output of the Vaaksetu extraction pipeline.
    All fields are optional to handle partial conversations where
    not all information is discussed.
    """
    patient_info: PatientInfo = Field(default_factory=PatientInfo)
    chief_complaint: Optional[str] = Field(
        None, description="Primary reason for the visit, in patient's words"
    )
    symptoms: List[Symptom] = Field(default_factory=list)
    vitals: Vitals = Field(default_factory=Vitals)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    current_medications: List[Medication] = Field(default_factory=list)
    lifestyle: Lifestyle = Field(default_factory=Lifestyle)
    provisional_diagnosis: List[str] = Field(default_factory=list)
    recommended_tests: List[str] = Field(default_factory=list)
    prescribed_medications: List[PrescribedMedication] = Field(default_factory=list)
    follow_up: FollowUp = Field(default_factory=FollowUp)
    conversation_summary: Optional[str] = None
    language_detected: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    def merge(self, other: "ClinicalRecord") -> "ClinicalRecord":
        """
        Merge another ClinicalRecord into this one.

        Useful for incremental extraction — merge partial results from
        multiple conversation turns into a single comprehensive record.
        Non-null/non-empty values from `other` take precedence.
        """
        merged_data = {}
        for field_name in self.model_fields:
            self_val = getattr(self, field_name)
            other_val = getattr(other, field_name)

            if isinstance(self_val, list) and isinstance(other_val, list):
                # Merge lists, avoiding exact duplicates
                existing_set = {
                    item.model_dump_json() if isinstance(item, BaseModel) else str(item)
                    for item in self_val
                }
                merged = list(self_val)
                for item in other_val:
                    key = item.model_dump_json() if isinstance(item, BaseModel) else str(item)
                    if key not in existing_set:
                        merged.append(item)
                merged_data[field_name] = merged

            elif isinstance(self_val, BaseModel) and isinstance(other_val, BaseModel):
                # Merge sub-models field by field
                sub_merged = {}
                for sub_field in self_val.model_fields:
                    sv = getattr(self_val, sub_field)
                    ov = getattr(other_val, sub_field)
                    if isinstance(sv, list) and isinstance(ov, list):
                        existing = set(sv)
                        sub_merged[sub_field] = sv + [x for x in ov if x not in existing]
                    else:
                        sub_merged[sub_field] = ov if ov is not None else sv
                merged_data[field_name] = type(self_val)(**sub_merged)

            else:
                merged_data[field_name] = other_val if other_val is not None else self_val

        return ClinicalRecord(**merged_data)

    def completeness_score(self) -> float:
        """
        Estimate how complete this record is (0.0 – 1.0).

        Checks key clinical fields and returns the fraction that are filled.
        """
        checks = [
            self.chief_complaint is not None,
            len(self.symptoms) > 0,
            any(v is not None for v in [
                self.vitals.blood_pressure_systolic,
                self.vitals.heart_rate_bpm,
                self.vitals.temperature_f,
            ]),
            len(self.medical_history.past_conditions) > 0,
            len(self.current_medications) > 0,
            len(self.provisional_diagnosis) > 0,
            len(self.recommended_tests) > 0,
            len(self.prescribed_medications) > 0,
            self.follow_up.when is not None,
            self.conversation_summary is not None,
        ]
        return sum(checks) / len(checks)
