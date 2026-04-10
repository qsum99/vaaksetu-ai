"""
Conversation Manager — Groq-powered medical dialogue engine.

Manages multi-turn medical conversations using Groq's ultra-fast
LLM inference API (LLaMA 3.3, Mixtral, etc.). Integrates symptom
logic, red flag detection, and context-aware follow-up questions
for structured patient intake.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .prompts import SYSTEM_PROMPT, SELECT_FOLLOW_UP_PROMPT
from .symptom_logic import SymptomLogic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq API configuration
# ---------------------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Available Groq models (for reference)
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Best quality, general purpose",
    "llama-3.1-8b-instant": "Fastest, good for simple follow-ups",
    "mixtral-8x7b-32768": "Good balance of speed and quality",
    "gemma2-9b-it": "Compact, efficient",
}


class ConversationManager:
    """
    Manages a medical conversation session with Groq's LLM API.

    Handles multi-turn dialogue, symptom tracking, red flag detection,
    and structured clinical summary generation. Each session maintains
    its own conversation history and symptom state.

    Groq provides ultra-fast inference (~500 tokens/s) making it ideal
    for real-time medical conversation flows.

    Usage::

        manager = ConversationManager(api_key="gsk_...")
        response = manager.start_session(patient_name="Ravi")
        print(response["message"])  # Greeting

        response = manager.process_message("I have chest pain for 3 days")
        print(response["message"])     # Follow-up question
        print(response["red_flags"])   # Any detected red flags
        print(response["symptoms"])    # Tracked symptoms

        summary = manager.generate_summary()
        print(summary["clinical_summary"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_DEFAULT_MODEL,
        base_url: str = GROQ_BASE_URL,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 30,
    ):
        """
        Args:
            api_key: Groq API key (starts with "gsk_"). Falls back to
                     GROQ_API_KEY env var.
            model: Groq model name (llama-3.3-70b-versatile,
                   llama-3.1-8b-instant, mixtral-8x7b-32768, etc.)
            base_url: Groq API base URL.
            temperature: Sampling temperature. Lower = more consistent.
            max_tokens: Max tokens per response.
            timeout: API call timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Session state
        self.session_id: str = str(uuid.uuid4())
        self.messages: List[dict] = []
        self.symptom_logic = SymptomLogic()
        self.started_at: Optional[datetime] = None
        self.patient_name: Optional[str] = None
        self.language: str = "en"

        # Groq client (lazy)
        self._client = None

        if not self.api_key:
            logger.warning(
                "No Groq API key. Set GROQ_API_KEY env var or pass api_key."
            )

    # ------------------------------------------------------------------
    # Client setup
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Lazy-load the Groq client (OpenAI-compatible SDK)."""
        if self._client is None:
            from groq import Groq

            self._client = Groq(
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        patient_name: Optional[str] = None,
        language: str = "en",
        context: Optional[str] = None,
    ) -> dict:
        """
        Start a new conversation session.

        Args:
            patient_name: Optional patient name for personalization.
            language: Conversation language code.
            context: Optional additional context (e.g. "follow-up visit").

        Returns:
            dict with: session_id, message (greeting), timestamp.
        """
        self.session_id = str(uuid.uuid4())
        self.started_at = datetime.now(timezone.utc)
        self.patient_name = patient_name
        self.language = language
        self.messages = []
        self.symptom_logic = SymptomLogic()

        # Initialize with system prompt
        system_content = SYSTEM_PROMPT
        if context:
            system_content += f"\n\nAdditional context: {context}"
        if language != "en":
            system_content += (
                f"\n\nThe patient prefers to communicate in "
                f"language code: {language}. Respond in that language "
                f"when appropriate, but keep medical terms clear."
            )

        self.messages.append({
            "role": "system",
            "content": system_content,
        })

        # Generate greeting
        greeting_prompt = "Greet the patient"
        if patient_name:
            greeting_prompt += f" named {patient_name}"
        greeting_prompt += (
            " warmly and ask what brings them in today. "
            "Keep it brief (2-3 sentences)."
        )

        greeting = self._call_groq(greeting_prompt)

        return {
            "session_id": self.session_id,
            "message": greeting,
            "timestamp": self.started_at.isoformat(),
            "type": "greeting",
        }

    def process_message(self, user_message: str) -> dict:
        """
        Process a patient message and generate an appropriate response.

        This is the main conversation loop entry point. It:
        1. Detects symptoms in the message
        2. Checks for red flags
        3. Determines the best follow-up question
        4. Generates a response via Groq

        Args:
            user_message: The patient's message text.

        Returns:
            dict with: message, symptoms, red_flags, assessment_status,
                       session_id, turn_number.
        """
        if not user_message or not user_message.strip():
            return {
                "message": "I didn't catch that. Could you please repeat?",
                "type": "error",
                "session_id": self.session_id,
            }

        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        # Step 1: Detect symptoms
        activated_chains = self.symptom_logic.detect_symptoms_in_text(user_message)

        # Step 2: Check red flags
        red_flags = self.symptom_logic.check_red_flags(user_message)

        # Step 3: Build context-aware prompt for Groq
        response_text = self._generate_response(user_message, red_flags)

        # Step 4: Track the question we asked
        next_q = self.symptom_logic.get_next_question()
        if next_q:
            self.symptom_logic.mark_question_asked(next_q["question"])

        # Add assistant response to history
        self.messages.append({"role": "assistant", "content": response_text})

        return {
            "session_id": self.session_id,
            "message": response_text,
            "type": "follow_up" if not red_flags else "red_flag",
            "symptoms": self.symptom_logic.identified_symptoms,
            "red_flags": [f["description"] for f in red_flags],
            "turn_number": self.symptom_logic.conversation_turns,
            "assessment_complete": self.symptom_logic.is_assessment_complete(),
            "remaining_questions": len(
                self.symptom_logic.get_remaining_questions()
            ),
        }

    def generate_summary(self) -> dict:
        """
        Generate a clinical summary of the conversation.

        Returns:
            dict with: clinical_summary, session_id, symptoms,
                       red_flags, conversation_turns.
        """
        summary_prompt = (
            "Based on our entire conversation, generate a structured "
            "clinical summary using this format:\n\n"
            "📋 **Clinical Summary**\n"
            "- **Chief Complaint**: ...\n"
            "- **Symptoms**: ... (with details)\n"
            "- **Duration**: ...\n"
            "- **Severity**: ...\n"
            "- **Relevant History**: ...\n"
            "- **Red Flags**: None / [list]\n"
            "- **Suggested Next Steps**: ...\n\n"
            "Be thorough and include all information gathered."
        )

        summary = self._call_groq(summary_prompt)

        return {
            "session_id": self.session_id,
            "clinical_summary": summary,
            "symptoms": self.symptom_logic.identified_symptoms,
            "red_flags": [
                f["description"] for f in self.symptom_logic.red_flags_detected
            ],
            "conversation_turns": self.symptom_logic.conversation_turns,
            "symptom_state": self.symptom_logic.get_state(),
        }

    def end_session(self) -> dict:
        """
        End the conversation session and return final state.

        Returns:
            dict with session summary and metadata.
        """
        summary = self.generate_summary()
        summary["ended_at"] = datetime.now(timezone.utc).isoformat()
        summary["started_at"] = (
            self.started_at.isoformat() if self.started_at else None
        )
        summary["total_messages"] = len(
            [m for m in self.messages if m["role"] != "system"]
        )
        return summary

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def _generate_response(
        self,
        user_message: str,
        red_flags: List[dict],
    ) -> str:
        """Generate a contextual response using Groq + symptom logic."""

        # If red flags detected, prioritize urgent response
        if red_flags:
            flag_descriptions = "; ".join(f["description"] for f in red_flags)
            urgent_prompt = (
                f"The patient said: \"{user_message}\"\n\n"
                f"⚠️ RED FLAGS DETECTED: {flag_descriptions}\n\n"
                f"Respond with:\n"
                f"1. Acknowledge their symptoms empathetically.\n"
                f"2. Clearly state this needs immediate medical attention.\n"
                f"3. Use the ⚠️ URGENT format.\n"
                f"4. Still ask any critical follow-up to help the doctor."
            )
            return self._call_groq(urgent_prompt)

        # Check if assessment is complete
        if self.symptom_logic.is_assessment_complete():
            return self._call_groq(
                f"The patient said: \"{user_message}\"\n\n"
                f"We have gathered enough information. Acknowledge their "
                f"response, let them know you have a good understanding of "
                f"their condition, and ask if there's anything else they'd "
                f"like to mention before you summarize."
            )

        # Get next follow-up question from symptom logic
        next_q = self.symptom_logic.get_next_question()

        if next_q:
            guide_prompt = (
                f"The patient said: \"{user_message}\"\n\n"
                f"Your next clinical question should be about: "
                f"{next_q['category']}\n"
                f"Suggested question: \"{next_q['question']}\"\n\n"
                f"Instructions:\n"
                f"1. First, briefly acknowledge what the patient said.\n"
                f"2. Then ask the follow-up question naturally.\n"
                f"3. Keep your response concise (2-3 sentences max).\n"
                f"4. Be empathetic and use simple language."
            )
        else:
            guide_prompt = (
                f"The patient said: \"{user_message}\"\n\n"
                f"Acknowledge their response and ask a relevant "
                f"follow-up question based on what they've told you."
            )

        return self._call_groq(guide_prompt)

    # ------------------------------------------------------------------
    # Groq API communication
    # ------------------------------------------------------------------

    def _call_groq(self, user_prompt: str) -> str:
        """
        Call the Groq API and return the response text.

        Groq provides ultra-fast inference (~500+ tokens/s) using
        custom LPU hardware, making it ideal for real-time
        medical conversations.
        """
        # Build messages for the call
        call_messages = self.messages.copy()
        call_messages.append({"role": "user", "content": user_prompt})

        t0 = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=call_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            elapsed = time.time() - t0

            # Groq returns usage stats including speed
            usage = response.usage
            tokens = getattr(usage, "total_tokens", "?") if usage else "?"

            logger.info(
                "Groq response — model=%s, tokens=%s, time=%.2fs",
                self.model, tokens, elapsed,
            )

            return content.strip()

        except Exception as e:
            logger.error("Groq API call failed: %s", e)
            # Fallback to rule-based response
            return self._fallback_response()

    def _fallback_response(self) -> str:
        """Generate a response without LLM when API is unavailable."""
        next_q = self.symptom_logic.get_next_question()
        if next_q:
            self.symptom_logic.mark_question_asked(next_q["question"])
            return (
                f"Thank you for sharing that. "
                f"{next_q['question']}"
            )

        if self.symptom_logic.is_assessment_complete():
            return (
                "Thank you for sharing all that information. "
                "I have a good understanding of your situation now. "
                "Is there anything else you'd like to mention?"
            )

        return (
            "I understand. Could you tell me a bit more about "
            "what you're experiencing?"
        )

    # ------------------------------------------------------------------
    # Session state management
    # ------------------------------------------------------------------

    def get_session_state(self) -> dict:
        """Export the full session state for persistence."""
        return {
            "session_id": self.session_id,
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "patient_name": self.patient_name,
            "language": self.language,
            "messages": [
                m for m in self.messages if m["role"] != "system"
            ],
            "symptom_state": self.symptom_logic.get_state(),
            "model": self.model,
        }

    def load_session_state(self, state: dict):
        """Restore session from a saved state."""
        self.session_id = state.get("session_id", str(uuid.uuid4()))
        started = state.get("started_at")
        if started:
            self.started_at = datetime.fromisoformat(started)
        self.patient_name = state.get("patient_name")
        self.language = state.get("language", "en")

        # Rebuild messages with system prompt
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.messages.extend(state.get("messages", []))

        # Restore symptom logic
        if "symptom_state" in state:
            self.symptom_logic.load_state(state["symptom_state"])

    def get_conversation_history(self) -> List[dict]:
        """Get human-readable conversation history (no system messages)."""
        return [
            {
                "role": m["role"],
                "content": m["content"],
            }
            for m in self.messages
            if m["role"] != "system"
        ]
