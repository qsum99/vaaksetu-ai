"""
Vaaksetu LLM Package — Groq-powered medical conversation engine.

Provides intelligent medical dialogue with symptom tracking, red flag
detection, and clinical summary generation using Groq's ultra-fast
LPU inference (LLaMA 3.3 70B, Mixtral, etc.).

Quick start::

    from ai_core.llm import ConversationManager

    manager = ConversationManager(api_key="gsk_...")
    greeting = manager.start_session(patient_name="Ravi")
    response = manager.process_message("I have severe headache for 2 days")
    summary = manager.generate_summary()
"""

from .conversation_manager import ConversationManager
from .symptom_logic import SymptomLogic

__all__ = [
    "ConversationManager",
    "SymptomLogic",
]
