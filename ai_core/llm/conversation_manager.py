"""
VaakSetu LLM Conversation Manager.
Sends conversation context to an OpenAI-compatible API and returns
follow-up questions + extracted clinical fields.
"""
import json
import os
import requests
from typing import List, Dict


def _get_config():
    """Lazy-load config to avoid import-time env var issues."""
    from config import Config
    return Config


def load_system_prompt() -> str:
    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            "You are VaakSetu, an AI clinical intake assistant. "
            "Extract symptoms, ask follow-ups, respond in JSON."
        )


def get_response(context: List[Dict[str, str]], transcript: str) -> dict:
    """
    Send the conversation context + new transcript to the LLM.
    Returns: { followup_question, extracted_fields, detected_symptoms }
    """
    cfg = _get_config()

    if not cfg.LLM_API_KEY:
        # Provide a mock response when no API key is configured
        print("[LLM] Warning: LLM_API_KEY not defined. Returning mock response.")
        return {
            "followup_question": "How long has this been happening?",
            "extracted_fields": {
                "symptoms": ["fever" if "fever" in transcript.lower() else "unknown"]
            },
            "detected_symptoms": ["fever"] if "fever" in transcript.lower() else [],
        }

    # Build messages
    messages = [{"role": "system", "content": load_system_prompt()}]
    messages.extend(context)
    messages.append({"role": "user", "content": transcript})

    headers = {
        "Authorization": f"Bearer {cfg.LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": cfg.LLM_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            f"{cfg.LLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()

        reply_content = result["choices"][0]["message"]["content"]
        parsed = json.loads(reply_content)
        return parsed

    except Exception as e:
        print(f"[LLM] API Error: {e}")
        return {
            "followup_question": "",
            "extracted_fields": {},
            "detected_symptoms": [],
        }
