"""
Symptom Logic Engine — Intelligent follow-up question routing.

Determines which follow-up questions to ask based on identified symptoms,
conversation history, and red flag detection. Works with or without the
LLM — can operate in pure rule-based mode.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from .prompts.follow_up_chains import FOLLOW_UP_CHAINS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symptom keyword → chain mapping
# ---------------------------------------------------------------------------

SYMPTOM_CHAIN_MAP = {
    # Chest pain variants
    "chest pain": "chest_pain",
    "heart pain": "chest_pain",
    "chest tightness": "chest_pain",
    "chest discomfort": "chest_pain",
    "seene mein dard": "chest_pain",        # Hindi
    "edeya novu": "chest_pain",              # Kannada

    # Fever variants
    "fever": "fever",
    "high temperature": "fever",
    "bukhar": "fever",                       # Hindi
    "jwara": "fever",                        # Kannada

    # Headache variants
    "headache": "headache",
    "head pain": "headache",
    "migraine": "headache",
    "sir dard": "headache",                  # Hindi
    "tale novu": "headache",                 # Kannada

    # Abdominal pain variants
    "stomach pain": "abdominal_pain",
    "abdominal pain": "abdominal_pain",
    "belly pain": "abdominal_pain",
    "pet dard": "abdominal_pain",            # Hindi
    "hottige novu": "abdominal_pain",        # Kannada

    # Cough variants
    "cough": "cough",
    "coughing": "cough",
    "khansi": "cough",                       # Hindi
    "kemmu": "cough",                        # Kannada

    # Joint pain variants
    "joint pain": "joint_pain",
    "knee pain": "joint_pain",
    "body pain": "joint_pain",
    "jodo ka dard": "joint_pain",            # Hindi

    # Breathing difficulty variants
    "breathing difficulty": "breathing_difficulty",
    "shortness of breath": "breathing_difficulty",
    "breathlessness": "breathing_difficulty",
    "saans ki taklif": "breathing_difficulty", # Hindi
    "ushirage": "breathing_difficulty",        # Kannada

    # Skin rash variants
    "rash": "skin_rash",
    "skin rash": "skin_rash",
    "itching": "skin_rash",
    "skin problem": "skin_rash",
}


# ---------------------------------------------------------------------------
# Red Flag Patterns (language-agnostic)
# ---------------------------------------------------------------------------

RED_FLAG_PATTERNS = [
    # Cardiac
    (r"chest\s*pain.*(?:arm|jaw|neck|back)", "Chest pain with radiation — possible cardiac event"),
    (r"(?:pain|ache).*(?:left\s*arm|jaw)", "Pain radiating to arm/jaw — rule out MI"),
    (r"cold\s*sweat|profuse\s*sweat", "Cold sweats — possible cardiac emergency"),

    # Neurological
    (r"worst\s*headache|thunderclap", "Thunderclap headache — rule out SAH"),
    (r"sudden\s*(?:numbness|weakness).*(?:one\s*side|left|right)", "Sudden unilateral weakness — rule out stroke"),
    (r"(?:loss\s*of|lost)\s*consciousness|fainted|blacked\s*out", "Loss of consciousness"),
    (r"confusion|disoriented|altered\s*mental", "Altered mental status"),
    (r"seizure|convulsion|fit", "Seizure activity"),

    # Respiratory
    (r"(?:can'?t|cannot|unable\s*to)\s*breathe", "Severe breathing difficulty"),
    (r"breathless\s*at\s*rest", "Breathlessness at rest"),
    (r"(?:blue|bluish)\s*(?:lips|fingers|nails)", "Cyanosis — oxygen deprivation"),

    # Fever
    (r"(?:fever|temperature).*(?:10[3-9]|1[1-9]\d)\s*(?:°?\s*f|fahrenheit)?", "High fever (≥103°F)"),
    (r"(?:fever|temperature).*(?:39\.[4-9]|[4-9]\d)\s*(?:°?\s*c|celsius)?", "High fever (≥39.4°C)"),
    (r"fever.*(?:more\s*than|over|>\s*)\s*3\s*days", "Prolonged fever >3 days"),

    # GI
    (r"blood.*(?:stool|vomit|spit)", "Blood in stool/vomit — GI bleed"),
    (r"(?:vomiting|throwing\s*up).*blood", "Hematemesis"),

    # General
    (r"uncontrolled\s*bleeding|won'?t\s*stop\s*bleeding", "Uncontrolled bleeding"),
    (r"suicid|self.?harm|end\s*(?:my|his|her)\s*life", "Suicidal ideation — immediate intervention"),
]


class SymptomLogic:
    """
    Manages symptom-aware follow-up question routing.

    Tracks which symptoms have been identified, which questions have
    been asked, and determines the optimal next question. Also performs
    real-time red flag detection.

    Usage::

        logic = SymptomLogic()
        logic.add_symptom("chest pain")

        next_q = logic.get_next_question()
        print(next_q)  # First chest pain follow-up question

        logic.mark_question_asked(next_q)
        flags = logic.check_red_flags("pain radiates to my left arm")
        print(flags)  # [("Chest pain with radiation — possible cardiac event", "high")]
    """

    def __init__(self):
        self.identified_symptoms: List[str] = []
        self.active_chains: List[str] = []
        self.questions_asked: List[str] = []
        self.red_flags_detected: List[dict] = []
        self.conversation_turns: int = 0
        self._asked_set: Set[str] = set()  # Fast lookup

    # ------------------------------------------------------------------
    # Symptom management
    # ------------------------------------------------------------------

    def add_symptom(self, symptom_text: str) -> Optional[str]:
        """
        Identify and register a symptom from patient text.

        Args:
            symptom_text: Raw text mentioning a symptom.

        Returns:
            The chain name activated, or None if no match.
        """
        symptom_lower = symptom_text.lower().strip()

        for keyword, chain_name in SYMPTOM_CHAIN_MAP.items():
            if keyword in symptom_lower:
                if chain_name not in self.active_chains:
                    self.active_chains.append(chain_name)
                    logger.info("Activated follow-up chain: %s", chain_name)

                if keyword not in self.identified_symptoms:
                    self.identified_symptoms.append(keyword)

                return chain_name

        # No specific match — use general chain
        if "general" not in self.active_chains:
            self.active_chains.append("general")
        self.identified_symptoms.append(symptom_lower)
        return "general"

    def detect_symptoms_in_text(self, text: str) -> List[str]:
        """
        Scan free text for any symptom keywords and register them.

        Returns:
            List of chain names activated.
        """
        text_lower = text.lower()
        activated = []

        for keyword, chain_name in SYMPTOM_CHAIN_MAP.items():
            if keyword in text_lower:
                result = self.add_symptom(keyword)
                if result and result not in activated:
                    activated.append(result)

        return activated

    # ------------------------------------------------------------------
    # Follow-up question logic
    # ------------------------------------------------------------------

    def get_next_question(self) -> Optional[dict]:
        """
        Determine the next follow-up question to ask.

        Returns:
            dict with keys: question, chain, index, category
            or None if all questions exhausted.
        """
        # Priority: red flag questions first, then active chains in order
        for chain_name in self.active_chains:
            chain = FOLLOW_UP_CHAINS.get(chain_name,
                                          FOLLOW_UP_CHAINS["general"])
            for i, question in enumerate(chain["questions"]):
                if question not in self._asked_set:
                    return {
                        "question": question,
                        "chain": chain_name,
                        "index": i,
                        "total": len(chain["questions"]),
                        "category": self._categorize_question(i, len(chain["questions"])),
                    }

        return None  # All questions exhausted

    def get_remaining_questions(self) -> List[dict]:
        """Get all remaining unanswered questions across active chains."""
        remaining = []
        for chain_name in self.active_chains:
            chain = FOLLOW_UP_CHAINS.get(chain_name,
                                          FOLLOW_UP_CHAINS["general"])
            for i, question in enumerate(chain["questions"]):
                if question not in self._asked_set:
                    remaining.append({
                        "question": question,
                        "chain": chain_name,
                        "index": i,
                    })
        return remaining

    def mark_question_asked(self, question: str):
        """Mark a question as asked."""
        self.questions_asked.append(question)
        self._asked_set.add(question)
        self.conversation_turns += 1

    def is_assessment_complete(self) -> bool:
        """Check if enough questions have been asked for a summary."""
        if not self.active_chains:
            return False

        # Complete if ≥60% of questions in all active chains are answered
        total = 0
        asked = 0
        for chain_name in self.active_chains:
            chain = FOLLOW_UP_CHAINS.get(chain_name,
                                          FOLLOW_UP_CHAINS["general"])
            total += len(chain["questions"])
            asked += sum(
                1 for q in chain["questions"] if q in self._asked_set
            )

        if total == 0:
            return False

        return (asked / total) >= 0.6 or self.conversation_turns >= 8

    # ------------------------------------------------------------------
    # Red flag detection
    # ------------------------------------------------------------------

    def check_red_flags(self, text: str) -> List[dict]:
        """
        Scan text for red flag patterns.

        Args:
            text: Patient's response text.

        Returns:
            List of red flag dicts with: pattern, description, urgency.
        """
        text_lower = text.lower()
        flags = []

        for pattern, description in RED_FLAG_PATTERNS:
            if re.search(pattern, text_lower):
                flag = {
                    "description": description,
                    "urgency": "high",
                    "matched_text": text[:100],
                }
                flags.append(flag)

                # Track globally
                if not any(f["description"] == description
                          for f in self.red_flags_detected):
                    self.red_flags_detected.append(flag)
                    logger.warning("🚨 RED FLAG: %s", description)

        return flags

    # ------------------------------------------------------------------
    # State export
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Export current state for serialization / context passing."""
        return {
            "identified_symptoms": self.identified_symptoms,
            "active_chains": self.active_chains,
            "questions_asked": self.questions_asked,
            "red_flags": self.red_flags_detected,
            "conversation_turns": self.conversation_turns,
            "assessment_complete": self.is_assessment_complete(),
            "remaining_questions": len(self.get_remaining_questions()),
        }

    def load_state(self, state: dict):
        """Restore state from a serialized dict."""
        self.identified_symptoms = state.get("identified_symptoms", [])
        self.active_chains = state.get("active_chains", [])
        self.questions_asked = state.get("questions_asked", [])
        self.red_flags_detected = state.get("red_flags", [])
        self.conversation_turns = state.get("conversation_turns", 0)
        self._asked_set = set(self.questions_asked)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize_question(index: int, total: int) -> str:
        """Categorize a question based on its position in the chain."""
        ratio = index / max(total, 1)
        if ratio < 0.3:
            return "symptom_detail"
        elif ratio < 0.6:
            return "associated_symptoms"
        elif ratio < 0.8:
            return "medical_history"
        else:
            return "lifestyle"
