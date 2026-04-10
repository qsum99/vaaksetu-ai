"""
Vaaksetu TTS Package — Text-to-speech for medical conversations.

Provides Indic language speech synthesis using Sarvam AI with
gTTS fallback for generating audio responses in Kannada, Hindi,
and other Indian languages.

Quick start::

    from ai_core.tts import SarvamTTS

    tts = SarvamTTS(api_key="your-sarvam-key")
    audio = tts.synthesize("ನಿಮ್ಮ ಆರೋಗ್ಯ ಹೇಗಿದೆ?", language="kn")

    # Or get base64 for web playback
    result = tts.synthesize_base64("How are you feeling?", language="en")
"""

from .sarvam_tts import SarvamTTS, TTSError

__all__ = [
    "SarvamTTS",
    "TTSError",
]
