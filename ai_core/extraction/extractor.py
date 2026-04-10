"""
Clinical Data Extractor — LLM-powered structured extraction.

Extracts structured medical data (Pydantic ClinicalRecord) from
doctor-patient conversation transcripts using an LLM with function-calling
or JSON-mode output. Supports incremental extraction across multiple
conversation turns.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union

from .models import ClinicalRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a medical data extraction assistant for the Vaaksetu platform.
Your task is to extract structured clinical information from doctor-patient
conversation transcripts.

RULES:
1. Extract ONLY information explicitly stated in the transcript.
2. Do NOT infer, guess, or fabricate any medical data.
3. Use exact medical terms when the patient or doctor uses them.
4. For symptoms, capture all OPQRST details when mentioned:
   Onset, Provocation/Palliation, Quality, Radiation, Severity, Time/Duration.
5. If information is ambiguous or unclear, set the field to null.
6. Support multilingual transcripts (English, Hindi, Kannada) — extract
   into English field values regardless of the source language.
7. Severity should be classified as "mild", "moderate", or "severe".
8. Duration should be in natural language (e.g. "3 days", "2 weeks").

OUTPUT FORMAT:
Return a valid JSON object matching the ClinicalRecord schema.
Do NOT include any explanation or markdown — return ONLY the JSON object.\
"""

EXTRACTION_PROMPT = """\
Extract structured clinical data from the following conversation transcript.

--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---

Language of transcript: {language}

Return a JSON object with these top-level keys:
- patient_info (name, age, gender, weight_kg, height_cm)
- chief_complaint (string)
- symptoms (array of objects with: name, body_region, severity, duration, \
frequency, onset, character, radiation, aggravating_factors, relieving_factors, \
associated_symptoms)
- vitals (blood_pressure_systolic, blood_pressure_diastolic, heart_rate_bpm, \
temperature_f, respiratory_rate, spo2_percent, blood_sugar_mg_dl)
- medical_history (past_conditions, surgeries, family_history, allergies)
- current_medications (array with: name, dosage, frequency, route)
- lifestyle (smoking, alcohol, diet, exercise, sleep, occupation)
- provisional_diagnosis (array of strings)
- recommended_tests (array of strings)
- prescribed_medications (array with: name, dosage, frequency, duration, instructions)
- follow_up (when, instructions, red_flags)
- conversation_summary (brief summary of the encounter)
- confidence_score (0.0 to 1.0, your confidence in the extraction accuracy)

Set any field to null or empty array if not mentioned in the transcript.\
"""

INCREMENTAL_PROMPT = """\
You have an existing clinical record from earlier parts of a conversation.
A new segment of the conversation has been transcribed. Extract any NEW
information from this segment and return ONLY the new/updated fields.

--- EXISTING RECORD ---
{existing_record}
--- END EXISTING RECORD ---

--- NEW TRANSCRIPT SEGMENT ---
{transcript}
--- END NEW SEGMENT ---

Return a JSON object with ONLY the fields that have new or updated information.
Do not repeat information already captured in the existing record.\
"""


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class ClinicalExtractor:
    """
    LLM-powered clinical data extractor.

    Takes conversation transcripts and produces structured ClinicalRecord
    objects using an LLM (OpenAI GPT-4 / GPT-3.5 or compatible API).

    Supports:
        - Full extraction from a complete transcript
        - Incremental extraction as new conversation turns arrive
        - Batch extraction from multiple transcripts

    Usage::

        extractor = ClinicalExtractor(api_key="sk-...")
        record = extractor.extract("Doctor: What brings you in today?\\n"
                                   "Patient: I have chest pain for 3 days.")
        print(record.chief_complaint)      # "chest pain for 3 days"
        print(record.symptoms[0].name)     # "chest pain"
        print(record.symptoms[0].duration) # "3 days"
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 60,
    ):
        """
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: LLM model name (e.g. "gpt-4o", "gpt-3.5-turbo").
            base_url: Custom API base URL (for Azure, local models, etc.)
            temperature: Sampling temperature. Low = more deterministic.
            max_tokens: Max output tokens for extraction.
            timeout: API call timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if not self.api_key:
            logger.warning(
                "No API key provided. Set OPENAI_API_KEY env var or pass "
                "api_key to ClinicalExtractor."
            )

        self._client = None

    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Full extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        transcript: str,
        language: str = "en",
    ) -> ClinicalRecord:
        """
        Extract a full ClinicalRecord from a conversation transcript.

        Args:
            transcript: The transcribed conversation text.
            language: Language code of the transcript.

        Returns:
            Validated ClinicalRecord with all extracted data.
        """
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript provided")
            return ClinicalRecord()

        logger.info(
            "Extracting clinical data from transcript (%d chars, lang=%s)",
            len(transcript), language,
        )

        user_prompt = EXTRACTION_PROMPT.format(
            transcript=transcript,
            language=language,
        )

        raw_json = self._call_llm(user_prompt)
        record = self._parse_to_record(raw_json, language)

        logger.info(
            "Extraction complete — %d symptoms, completeness=%.0f%%",
            len(record.symptoms),
            record.completeness_score() * 100,
        )
        return record

    # ------------------------------------------------------------------
    # Incremental extraction
    # ------------------------------------------------------------------

    def extract_incremental(
        self,
        new_transcript: str,
        existing_record: ClinicalRecord,
        language: str = "en",
    ) -> ClinicalRecord:
        """
        Extract new information from a conversation segment and merge
        it into an existing record.

        Args:
            new_transcript: New segment of conversation text.
            existing_record: Previously extracted ClinicalRecord.
            language: Language code.

        Returns:
            Merged ClinicalRecord with old + new data.
        """
        if not new_transcript or not new_transcript.strip():
            return existing_record

        logger.info(
            "Incremental extraction — existing record has %d symptoms, "
            "new segment has %d chars",
            len(existing_record.symptoms),
            len(new_transcript),
        )

        existing_json = existing_record.model_dump_json(indent=2, exclude_none=True)

        user_prompt = INCREMENTAL_PROMPT.format(
            existing_record=existing_json,
            transcript=new_transcript,
        )

        raw_json = self._call_llm(user_prompt)
        new_record = self._parse_to_record(raw_json, language)

        # Merge new into existing
        merged = existing_record.merge(new_record)

        logger.info(
            "Merge complete — now %d symptoms, completeness=%.0f%%",
            len(merged.symptoms),
            merged.completeness_score() * 100,
        )
        return merged

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        transcripts: List[str],
        language: str = "en",
    ) -> List[ClinicalRecord]:
        """
        Extract clinical records from multiple transcripts.

        Args:
            transcripts: List of transcript strings.
            language: Language code.

        Returns:
            List of ClinicalRecord objects.
        """
        results = []
        for i, transcript in enumerate(transcripts):
            logger.info("Batch extraction %d/%d", i + 1, len(transcripts))
            record = self.extract(transcript, language)
            results.append(record)
        return results

    # ------------------------------------------------------------------
    # Rule-based fallback extractor (no LLM needed)
    # ------------------------------------------------------------------

    def extract_rules_based(
        self,
        transcript: str,
        language: str = "en",
    ) -> ClinicalRecord:
        """
        Simple rule-based extraction as a fallback when LLM is unavailable.

        Uses keyword matching and regex patterns for common medical terms.
        Accuracy is lower than LLM extraction but requires no API key.

        Args:
            transcript: Conversation transcript text.
            language: Language code.

        Returns:
            ClinicalRecord with rule-extracted data.
        """
        import re
        from .models import (
            PatientInfo, Symptom, Vitals, MedicalHistory,
            Medication, Lifestyle, FollowUp,
        )

        logger.info("Rule-based extraction from %d chars", len(transcript))
        text = transcript.lower()

        # --- Chief complaint ---
        chief = None
        complaint_patterns = [
            r"(?:complaint|problem|issue|reason)[\s:]+(.+?)[\.\n]",
            r"(?:i have|i am having|i feel|i've been having)\s+(.+?)[\.\n]",
            r"(?:suffering from|experiencing)\s+(.+?)[\.\n]",
        ]
        for pattern in complaint_patterns:
            match = re.search(pattern, text)
            if match:
                chief = match.group(1).strip()
                break

        # --- Symptoms (keyword matching) ---
        symptom_keywords = {
            "chest pain": "chest",
            "headache": "head",
            "fever": "systemic",
            "cough": "respiratory",
            "cold": "respiratory",
            "body pain": "systemic",
            "back pain": "back",
            "stomach pain": "abdomen",
            "abdominal pain": "abdomen",
            "nausea": "gastrointestinal",
            "vomiting": "gastrointestinal",
            "diarrhea": "gastrointestinal",
            "dizziness": "neurological",
            "fatigue": "systemic",
            "weakness": "systemic",
            "breathlessness": "respiratory",
            "shortness of breath": "respiratory",
            "joint pain": "musculoskeletal",
            "sore throat": "throat",
            "palpitations": "cardiac",
            "swelling": None,
            "rash": "skin",
            "itching": "skin",
            "burning sensation": None,
            "loss of appetite": "gastrointestinal",
            "weight loss": "systemic",
            "insomnia": "neurological",
        }
        symptoms = []
        for symptom_name, region in symptom_keywords.items():
            if symptom_name in text:
                # Try to find severity
                severity = None
                for sev in ["severe", "moderate", "mild"]:
                    if re.search(rf"{sev}\s+{re.escape(symptom_name)}", text):
                        severity = sev
                        break
                    if re.search(rf"{re.escape(symptom_name)}.*?{sev}", text):
                        severity = sev
                        break

                # Try to find duration
                duration = None
                dur_match = re.search(
                    rf"{re.escape(symptom_name)}.*?(?:for|since|from)\s+"
                    rf"(\d+\s*(?:day|week|month|hour|year)s?)",
                    text,
                )
                if dur_match:
                    duration = dur_match.group(1).strip()

                symptoms.append(Symptom(
                    name=symptom_name,
                    body_region=region,
                    severity=severity,
                    duration=duration,
                ))

        # --- Vitals (regex) ---
        vitals = Vitals()
        bp_match = re.search(r"(?:bp|blood pressure)[\s:]*(\d{2,3})\s*/\s*(\d{2,3})", text)
        if bp_match:
            vitals.blood_pressure_systolic = int(bp_match.group(1))
            vitals.blood_pressure_diastolic = int(bp_match.group(2))

        hr_match = re.search(r"(?:heart rate|pulse|hr)[\s:]*(\d{2,3})", text)
        if hr_match:
            vitals.heart_rate_bpm = int(hr_match.group(1))

        temp_match = re.search(r"(?:temperature|temp)[\s:]*(\d{2,3}\.?\d*)\s*(?:f|°f)?", text)
        if temp_match:
            vitals.temperature_f = float(temp_match.group(1))

        spo2_match = re.search(r"(?:spo2|oxygen|saturation)[\s:]*(\d{2,3})[\s]*%?", text)
        if spo2_match:
            vitals.spo2_percent = float(spo2_match.group(1))

        sugar_match = re.search(r"(?:sugar|glucose|blood sugar)[\s:]*(\d{2,3})", text)
        if sugar_match:
            vitals.blood_sugar_mg_dl = float(sugar_match.group(1))

        # --- Age / Gender ---
        patient_info = PatientInfo()
        age_match = re.search(r"(\d{1,3})\s*(?:year|yr)s?\s*(?:old)?", text)
        if age_match:
            patient_info.age = int(age_match.group(1))

        if "female" in text or "woman" in text or "lady" in text:
            patient_info.gender = "female"
        elif "male" in text or " man " in text:
            patient_info.gender = "male"

        # --- Allergies ---
        medical_history = MedicalHistory()
        allergy_match = re.search(r"allerg(?:y|ic|ies)\s+(?:to\s+)?(.+?)[\.\n,]", text)
        if allergy_match:
            allergies = [a.strip() for a in allergy_match.group(1).split(" and ")]
            medical_history.allergies = allergies

        # --- Conditions ---
        condition_keywords = [
            "diabetes", "hypertension", "asthma", "thyroid", "cholesterol",
            "arthritis", "migraine", "anemia", "depression", "anxiety",
        ]
        for cond in condition_keywords:
            if cond in text:
                medical_history.past_conditions.append(cond)

        return ClinicalRecord(
            patient_info=patient_info,
            chief_complaint=chief,
            symptoms=symptoms,
            vitals=vitals,
            medical_history=medical_history,
            language_detected=language,
            confidence_score=0.4,  # Lower confidence for rule-based
        )

    # ------------------------------------------------------------------
    # LLM communication
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM and return the raw JSON string response."""
        t0 = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            elapsed = time.time() - t0

            logger.info(
                "LLM call complete — model=%s, tokens=%s, time=%.1fs",
                self.model,
                getattr(response.usage, "total_tokens", "?"),
                elapsed,
            )
            return content

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise ExtractionError(f"LLM call failed: {e}") from e

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_to_record(self, raw_json: str, language: str) -> ClinicalRecord:
        """Parse LLM JSON output into a validated ClinicalRecord."""
        try:
            # Strip any markdown code fences the LLM might add
            cleaned = raw_json.strip()
            if cleaned.startswith("```"):
                # Remove opening fence
                first_newline = cleaned.index("\n")
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)

            # Inject language if not set
            if "language_detected" not in data or not data["language_detected"]:
                data["language_detected"] = language

            record = ClinicalRecord.model_validate(data)
            return record

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM output as JSON: %s", e)
            logger.debug("Raw output: %s", raw_json[:500])
            raise ExtractionError(f"Invalid JSON from LLM: {e}") from e

        except Exception as e:
            logger.error("Failed to validate extracted data: %s", e)
            raise ExtractionError(f"Validation error: {e}") from e


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    """Raised when clinical data extraction fails."""
    pass
