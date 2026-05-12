"""
VaakSetu — Centralized Configuration
Loads all environment variables once at import time.
"""
import os
from dotenv import load_dotenv

# Load .env FIRST, before anything else reads os.getenv
load_dotenv()


class Config:
    """Application-wide configuration pulled from environment."""

    # ── Flask ──────────────────────────────────────────────
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "vaaksetu-dev-secret-key-change-in-prod")
    DEBUG = os.getenv("FLASK_DEBUG", "true").lower() in ("1", "true", "yes")

    # ── Database ───────────────────────────────────────────
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vaaksetu.db")
    SQLALCHEMY_CONNECT_ARGS = (
        {"check_same_thread": False} if "sqlite" in os.getenv("DATABASE_URL", "sqlite") else {}
    )

    # ── LLM (OpenAI-compatible) ────────────────────────────
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

    # ── Sarvam TTS ─────────────────────────────────────────
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

    # ── Server ─────────────────────────────────────────────
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
