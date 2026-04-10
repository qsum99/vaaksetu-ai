"""
Language Detector — Routes audio to the correct ASR engine.

Detects the spoken language from an audio sample and dispatches to either
WhisperASR (English/Hindi) or IndicWhisperASR (Kannada/Indic) for
full transcription. Supports code-mixed detection.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language → engine mapping
# ---------------------------------------------------------------------------
WHISPER_LANGUAGES = {"en", "hi"}          # Handled by OpenAI Whisper
INDIC_LANGUAGES = {"kn", "ta", "te", "ml", "bn", "gu", "mr"}  # IndicWhisper


class LanguageDetector:
    """
    Detects spoken language from audio and routes to the appropriate
    ASR engine (Whisper vs IndicWhisper).

    The detector uses Whisper's built-in language identification on the
    first 30 seconds of audio to classify the language, then dispatches
    the full audio to the correct model.

    Usage::

        detector = LanguageDetector()
        result = detector.detect_and_transcribe("patient_audio.wav")
        print(result["language"])   # "kn"
        print(result["text"])       # Kannada transcription
        print(result["engine"])     # "indic_whisper"
    """

    def __init__(
        self,
        whisper_model: str = "large-v3",
        indic_model: str = "ai4bharat/indicwhisper-hindi-large-v2",
        device: Optional[str] = None,
        default_language: Optional[str] = None,
    ):
        """
        Args:
            whisper_model: Whisper model name for English/Hindi.
            indic_model: HuggingFace model ID for Indic languages.
            device: "cuda" / "cpu". Auto-detected if None.
            default_language: Force a language instead of auto-detecting.
        """
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model = whisper_model
        self.indic_model = indic_model
        self.default_language = default_language

        # Lazy-loaded engine instances
        self._whisper_asr = None
        self._indic_asr = None

    # ------------------------------------------------------------------
    # Engine accessors (lazy)
    # ------------------------------------------------------------------

    @property
    def whisper_asr(self):
        if self._whisper_asr is None:
            from .whisper_asr import WhisperASR
            self._whisper_asr = WhisperASR(
                model_name=self.whisper_model,
                device=self.device,
            )
        return self._whisper_asr

    @property
    def indic_asr(self):
        if self._indic_asr is None:
            from .indic_whisper import IndicWhisperASR
            self._indic_asr = IndicWhisperASR(
                model_id=self.indic_model,
                device=f"{self.device}:0" if self.device == "cuda" else self.device,
            )
        return self._indic_asr

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def detect_language(self, audio_path: Union[str, Path]) -> dict:
        """
        Detect the spoken language in an audio file using Whisper's
        built-in language identification.

        Args:
            audio_path: Path to audio file.

        Returns:
            dict with keys:
                - language (str): ISO 639-1 code (e.g. "en", "hi", "kn").
                - confidence (float): Detection confidence [0, 1].
                - probabilities (dict): Top-5 language probabilities.
                - detection_time_s (float): Time taken for detection.
        """
        import torch
        import whisper

        audio_path = str(audio_path)
        logger.info("Detecting language for: %s", audio_path)

        t0 = time.time()

        # Load audio and extract mel spectrogram (first 30s)
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # Use Whisper's language detection
        model = self.whisper_asr.model
        _, probs = model.detect_language(mel)

        detection_time = time.time() - t0

        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        detected_lang = sorted_probs[0][0]
        confidence = sorted_probs[0][1]

        result = {
            "language": detected_lang,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in sorted_probs[:5]},
            "detection_time_s": round(detection_time, 2),
        }

        logger.info(
            "Language detected: %s (confidence=%.2f%%) in %.1fs",
            detected_lang,
            confidence * 100,
            detection_time,
        )
        return result

    def detect_language_from_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> dict:
        """Detect language from raw audio bytes."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.detect_language(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Detect + Transcribe (main entry point)
    # ------------------------------------------------------------------

    def detect_and_transcribe(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        audio_bytes: Optional[bytes] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: int = 16_000,
        force_language: Optional[str] = None,
    ) -> dict:
        """
        Detect language and transcribe using the appropriate engine.

        Provide exactly one audio source: audio_path, audio_bytes, or
        audio_array.

        Args:
            audio_path: Path to audio file on disk.
            audio_bytes: Raw audio bytes.
            audio_array: Numpy audio array (float32).
            sample_rate: Sample rate for numpy arrays.
            force_language: Override auto-detection with this language.

        Returns:
            dict with all transcription fields plus:
                - engine (str): "whisper" or "indic_whisper"
                - detected_language (str): What the detector found.
        """
        # Resolve the language
        language = force_language or self.default_language

        if language is None:
            # Need to detect — requires file path
            if audio_path:
                det = self.detect_language(audio_path)
            elif audio_bytes:
                det = self.detect_language_from_bytes(audio_bytes)
            elif audio_array is not None:
                # Write numpy to temp file for detection
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_array, sample_rate)
                    tmp_path = tmp.name
                try:
                    det = self.detect_language(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                raise ValueError(
                    "Provide one of: audio_path, audio_bytes, or audio_array"
                )

            language = det["language"]
        else:
            det = {"language": language, "confidence": 1.0}

        # Route to the correct engine
        engine_name = self._select_engine(language)
        logger.info(
            "Routing language '%s' → engine '%s'", language, engine_name
        )

        if engine_name == "indic_whisper":
            engine = self.indic_asr
            engine.language = language
        else:
            engine = self.whisper_asr
            engine.language = language

        # Transcribe
        if audio_path:
            result = engine.transcribe_file(audio_path)
        elif audio_bytes:
            result = engine.transcribe_bytes(audio_bytes)
        elif audio_array is not None:
            result = engine.transcribe_numpy(audio_array, sample_rate)
        else:
            raise ValueError(
                "Provide one of: audio_path, audio_bytes, or audio_array"
            )

        # Augment result with detection info
        result["engine"] = engine_name
        result["detected_language"] = det["language"]
        result["detection_confidence"] = det.get("confidence", 1.0)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _select_engine(self, language: str) -> str:
        """Pick the best engine for a given language code."""
        if language in INDIC_LANGUAGES:
            return "indic_whisper"
        elif language in WHISPER_LANGUAGES:
            return "whisper"
        else:
            # Fallback to Whisper for unknown languages
            logger.warning(
                "Language '%s' not in known set, falling back to Whisper",
                language,
            )
            return "whisper"

    def get_supported_languages(self) -> dict:
        """Return all supported languages grouped by engine."""
        return {
            "whisper": sorted(WHISPER_LANGUAGES),
            "indic_whisper": sorted(INDIC_LANGUAGES),
        }
