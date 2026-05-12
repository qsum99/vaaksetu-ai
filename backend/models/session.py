from pydantic import BaseModel
from typing import List, Dict, Optional

class SessionState(BaseModel):
    session_id: str
    context: List[Dict[str, str]] = []  # List of { role: ..., content: ... }
    transcript: str = ""
    detected_languages: List[str] = []
    extracted_fields: dict = {}
    detected_symptoms: List[str] = []
