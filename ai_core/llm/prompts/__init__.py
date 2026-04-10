"""
Prompts package — All prompt templates for the Vaaksetu LLM engine.
"""

from .system_prompt import SYSTEM_PROMPT
from .follow_up_chains import FOLLOW_UP_CHAINS, SELECT_FOLLOW_UP_PROMPT

__all__ = [
    "SYSTEM_PROMPT",
    "FOLLOW_UP_CHAINS",
    "SELECT_FOLLOW_UP_PROMPT",
]
