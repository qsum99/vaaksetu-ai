"""
Clinical data extractor.
Combines LLM-extracted fields with regex fallbacks, validates
against the JSON schema, and returns a clean dict.
"""
import json
import os
import re
from jsonschema import validate, ValidationError
from datetime import date


def load_schema() -> dict:
    current_dir = os.path.dirname(__file__)
    schema_path = os.path.join(current_dir, "clinical_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract(transcript: str, llm_fields: dict, language_mix: list) -> dict:
    """
    Build a clinical record dict from LLM output + regex fallbacks.
    Returns a plain dict (decoupled from backend models).
    """
    extracted = {
        "patient_name": llm_fields.get("name") or "Unknown Patient",
        "age": llm_fields.get("age"),
        "gender": llm_fields.get("gender") or "other",
        "village": llm_fields.get("village") or "Unknown Village",
        "chief_complaint": (
            llm_fields.get("chief_complaint")
            or " ".join(llm_fields.get("symptoms", []))
        ),
        "symptoms": llm_fields.get("symptoms", []),
        "duration": llm_fields.get("duration") or "Unknown",
        "vitals": llm_fields.get("vitals", {}),
        "medical_history": llm_fields.get("medical_history"),
        "diagnosis_notes": None,
        "followup_date": None,
        "raw_transcript": transcript,
        "language_mix": language_mix,
        "created_at": str(date.today()),
    }

    # ── Regex fallbacks ──────────────────────────────────
    if not extracted["age"]:
        age_match = re.search(r"(\d+)\s*years?", transcript, re.IGNORECASE)
        extracted["age"] = int(age_match.group(1)) if age_match else 0

    if not extracted["vitals"]:
        vitals = {}
        bp_match = re.search(r"(\d{2,3}/\d{2,3})", transcript)
        temp_match = re.search(r"(\d{2,3})\s*fever", transcript, re.IGNORECASE)
        if bp_match:
            vitals["bp"] = bp_match.group(1)
        if temp_match:
            vitals["temp"] = temp_match.group(1) + "F"
        extracted["vitals"] = vitals

    # ── Schema validation ────────────────────────────────
    schema = load_schema()
    try:
        validate(instance=extracted, schema=schema)
    except ValidationError as e:
        print(f"[extractor] Validation Warning: {e.message}")

    return extracted
