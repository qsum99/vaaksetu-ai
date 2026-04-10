"""
Vaaksetu ASR Package — Speech recognition for medical conversations.

Provides three engines:
    - WhisperASR: OpenAI Whisper (English + Hindi)
    - IndicWhisperASR: AI4Bharat IndicWhisper (Kannada + Indic)
    - LanguageDetector: Auto-detect language and route to correct engine

Quick start::

    from ai_core.asr import LanguageDetector

    detector = LanguageDetector()
    result = detector.detect_and_transcribe(audio_path="recording.wav")
    print(result["text"], result["language"], result["engine"])
"""

from .whisper_asr import WhisperASR
from .indic_whisper import IndicWhisperASR
from .lang_detector import LanguageDetector

__all__ = [
    "WhisperASR",
    "IndicWhisperASR",
    "LanguageDetector",
]
