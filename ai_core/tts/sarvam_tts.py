"""
Sarvam TTS Engine — Indic text-to-speech for Vaaksetu.

Uses Sarvam AI's TTS API for high-quality Kannada, Hindi, and other
Indic language speech synthesis. Includes a gTTS fallback for when
the Sarvam API is unavailable.
"""

import base64
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sarvam AI API configuration
# ---------------------------------------------------------------------------
SARVAM_API_URL = "https://api.sarvam.ai/text-to-speech"

# Sarvam voice options per language
SARVAM_VOICES = {
    "kn": {"voice": "arvind", "name": "Kannada Male"},
    "hi": {"voice": "amol", "name": "Hindi Male"},
    "ta": {"voice": "kumar", "name": "Tamil Male"},
    "te": {"voice": "ravi", "name": "Telugu Male"},
    "ml": {"voice": "anand", "name": "Malayalam Male"},
    "bn": {"voice": "soham", "name": "Bengali Male"},
    "gu": {"voice": "neel", "name": "Gujarati Male"},
    "mr": {"voice": "aditya", "name": "Marathi Male"},
    "en": {"voice": "meera", "name": "English Female"},
}

# Language code → Sarvam language code mapping
SARVAM_LANG_MAP = {
    "kn": "kn-IN",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "mr": "mr-IN",
    "en": "en-IN",
}


class SarvamTTS:
    """
    Text-to-Speech engine using Sarvam AI for Indic languages.

    Generates natural-sounding speech for Kannada, Hindi, and other
    Indian languages. Includes gTTS fallback for reliability.

    Usage::

        tts = SarvamTTS(api_key="your-sarvam-key")
        audio_bytes = tts.synthesize("ನಿಮ್ಮ ಆರೋಗ್ಯ ಹೇಗಿದೆ?", language="kn")

        # Save to file
        tts.synthesize_to_file("Hello, how are you?", "output.wav", language="en")

        # Get base64 for web playback
        b64 = tts.synthesize_base64("आपकी तबीयत कैसी है?", language="hi")
    """

    SUPPORTED_LANGUAGES = set(SARVAM_LANG_MAP.keys())

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_language: str = "kn",
        speech_rate: float = 1.0,
        pitch: float = 1.0,
        timeout: int = 30,
        use_fallback: bool = True,
    ):
        """
        Args:
            api_key: Sarvam AI API key. Falls back to SARVAM_API_KEY env var.
            default_language: Default language code (ISO 639-1).
            speech_rate: Speech speed multiplier (0.5 = slow, 2.0 = fast).
            pitch: Pitch multiplier (0.5 = low, 2.0 = high).
            timeout: API request timeout in seconds.
            use_fallback: If True, fall back to gTTS when Sarvam fails.
        """
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.default_language = default_language
        self.speech_rate = speech_rate
        self.pitch = pitch
        self.timeout = timeout
        self.use_fallback = use_fallback

        if not self.api_key:
            logger.warning(
                "No Sarvam API key. Set SARVAM_API_KEY env var. "
                "Will use gTTS fallback."
            )

    # ------------------------------------------------------------------
    # Main synthesis methods
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> bytes:
        """
        Convert text to speech audio bytes.

        Args:
            text: Text to synthesize.
            language: Language code (e.g. "kn", "hi", "en").
            voice: Override voice name.

        Returns:
            Audio bytes (WAV format).
        """
        language = language or self.default_language

        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return b""

        logger.info(
            "Synthesizing TTS — lang=%s, text_len=%d",
            language, len(text),
        )

        # Try Sarvam API first
        if self.api_key:
            try:
                return self._synthesize_sarvam(text, language, voice)
            except Exception as e:
                logger.error("Sarvam TTS failed: %s", e)
                if not self.use_fallback:
                    raise

        # Fallback to gTTS
        logger.info("Using gTTS fallback")
        return self._synthesize_gtts(text, language)

    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> str:
        """
        Convert text to speech and save to a file.

        Args:
            text: Text to synthesize.
            output_path: Output file path.
            language: Language code.
            voice: Override voice name.

        Returns:
            Absolute path to the saved audio file.
        """
        audio_bytes = self.synthesize(text, language, voice)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)

        logger.info("Audio saved to: %s (%d bytes)", output_path, len(audio_bytes))
        return str(output_path.resolve())

    def synthesize_base64(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> dict:
        """
        Convert text to speech and return as base64-encoded string.

        Convenient for embedding audio directly in JSON API responses
        or playing in the browser via data URLs.

        Args:
            text: Text to synthesize.
            language: Language code.
            voice: Override voice name.

        Returns:
            dict with: audio_base64, content_type, language, text_length.
        """
        audio_bytes = self.synthesize(text, language, voice)

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "content_type": "audio/wav",
            "language": language or self.default_language,
            "text_length": len(text),
            "audio_size_bytes": len(audio_bytes),
        }

    # ------------------------------------------------------------------
    # Chunked synthesis (for long text)
    # ------------------------------------------------------------------

    def synthesize_long_text(
        self,
        text: str,
        language: Optional[str] = None,
        max_chunk_length: int = 500,
    ) -> bytes:
        """
        Synthesize long text by splitting into chunks.

        Sarvam API has text length limits, so this method splits
        long text at sentence boundaries and concatenates the audio.

        Args:
            text: Long text to synthesize.
            language: Language code.
            max_chunk_length: Max characters per chunk.

        Returns:
            Combined audio bytes.
        """
        chunks = self._split_text(text, max_chunk_length)
        logger.info("Long text split into %d chunks", len(chunks))

        audio_parts = []
        for i, chunk in enumerate(chunks):
            logger.info("Synthesizing chunk %d/%d", i + 1, len(chunks))
            audio = self.synthesize(chunk, language)
            audio_parts.append(audio)

        # Simple concatenation (works for WAV with same params)
        return self._concatenate_audio(audio_parts)

    # ------------------------------------------------------------------
    # Sarvam AI API
    # ------------------------------------------------------------------

    def _synthesize_sarvam(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
    ) -> bytes:
        """Call Sarvam AI TTS API."""
        t0 = time.time()

        lang_code = SARVAM_LANG_MAP.get(language, f"{language}-IN")
        if voice is None:
            voice_info = SARVAM_VOICES.get(language, SARVAM_VOICES["en"])
            voice = voice_info["voice"]

        payload = {
            "inputs": [text],
            "target_language_code": lang_code,
            "speaker": voice,
            "pitch": self.pitch,
            "pace": self.speech_rate,
            "loudness": 1.5,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": "bulbul:v1",
        }

        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key,
        }

        response = requests.post(
            SARVAM_API_URL,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise TTSError(
                f"Sarvam API error {response.status_code}: {response.text}"
            )

        result = response.json()
        elapsed = time.time() - t0

        # Sarvam returns base64-encoded audio
        audios = result.get("audios", [])
        if not audios:
            raise TTSError("No audio returned from Sarvam API")

        audio_b64 = audios[0]
        audio_bytes = base64.b64decode(audio_b64)

        logger.info(
            "Sarvam TTS complete — lang=%s, voice=%s, "
            "audio=%d bytes, time=%.2fs",
            language, voice, len(audio_bytes), elapsed,
        )

        return audio_bytes

    # ------------------------------------------------------------------
    # gTTS Fallback
    # ------------------------------------------------------------------

    def _synthesize_gtts(self, text: str, language: str) -> bytes:
        """Fallback TTS using Google's gTTS (free, lower quality)."""
        from gtts import gTTS

        t0 = time.time()

        # gTTS language mapping
        gtts_lang_map = {
            "kn": "kn",
            "hi": "hi",
            "ta": "ta",
            "te": "te",
            "ml": "ml",
            "bn": "bn",
            "gu": "gu",
            "mr": "mr",
            "en": "en",
        }
        gtts_lang = gtts_lang_map.get(language, "en")

        tts = gTTS(text=text, lang=gtts_lang, slow=False)

        # Write to buffer
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        audio_bytes = buf.getvalue()

        elapsed = time.time() - t0
        logger.info(
            "gTTS fallback complete — lang=%s, audio=%d bytes, time=%.2fs",
            language, len(audio_bytes), elapsed,
        )

        return audio_bytes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_text(text: str, max_length: int = 500) -> list:
        """Split text at sentence boundaries for chunked synthesis."""
        import re

        if len(text) <= max_length:
            return [text]

        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    @staticmethod
    def _concatenate_audio(audio_parts: list) -> bytes:
        """Concatenate multiple audio byte segments."""
        if len(audio_parts) == 1:
            return audio_parts[0]

        # Simple concatenation — for production, use pydub or ffmpeg
        return b"".join(audio_parts)

    def get_available_voices(self) -> dict:
        """Return available voices grouped by language."""
        return {
            lang: info
            for lang, info in SARVAM_VOICES.items()
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TTSError(Exception):
    """Raised when text-to-speech synthesis fails."""
    pass
