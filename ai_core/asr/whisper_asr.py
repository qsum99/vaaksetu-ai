"""
Whisper ASR Engine — English + Hindi speech recognition.

Uses OpenAI's Whisper (large-v3 by default) for high-accuracy transcription
of English and Hindi medical conversations. Supports file-based and
in-memory (numpy array) transcription with word-level timestamps.
"""

import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded global model cache (one model at a time to conserve VRAM)
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def _get_model(model_name: str = "large-v3", device: Optional[str] = None):
    """Load and cache a Whisper model. Thread-safe via GIL for CPython."""
    import torch
    import whisper

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = f"{model_name}@{device}"
    if cache_key not in _model_cache:
        logger.info("Loading Whisper model '%s' on %s …", model_name, device)
        t0 = time.time()
        _model_cache[cache_key] = whisper.load_model(model_name, device=device)
        logger.info("Model loaded in %.1f s", time.time() - t0)

    return _model_cache[cache_key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class WhisperASR:
    """
    Whisper-based Automatic Speech Recognition for English and Hindi.

    Designed for the Vaaksetu medical platform — captures doctor-patient
    conversations with high accuracy using OpenAI's Whisper large-v3.

    Usage::

        asr = WhisperASR(model_name="large-v3")
        result = asr.transcribe_file("patient_audio.wav")
        print(result["text"])
        print(result["language"])
    """

    SUPPORTED_LANGUAGES = {"en", "hi"}  # English + Hindi

    def __init__(
        self,
        model_name: str = "large-v3",
        device: Optional[str] = None,
        language: Optional[str] = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
    ):
        """
        Args:
            model_name: Whisper model size — tiny, base, small, medium,
                        large, large-v2, large-v3.
            device: "cuda" or "cpu". Auto-detected if None.
            language: ISO 639-1 code to force (e.g. "en", "hi").
                      None = auto-detect.
            beam_size: Beam search width. Higher = slower but more accurate.
            word_timestamps: Emit per-word timestamps in the output.
        """
        import torch

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.beam_size = beam_size
        self.word_timestamps = word_timestamps
        self._model = None  # lazy load

    @property
    def model(self):
        """Lazy-load the underlying Whisper model."""
        if self._model is None:
            self._model = _get_model(self.model_name, self.device)
        return self._model

    # ------------------------------------------------------------------
    # Transcription methods
    # ------------------------------------------------------------------

    def transcribe_file(self, audio_path: Union[str, Path]) -> dict:
        """
        Transcribe an audio file on disk.

        Args:
            audio_path: Path to a WAV / MP3 / M4A / FLAC / OGG file.

        Returns:
            dict with keys:
                - text (str): Full transcription.
                - language (str): Detected / forced language code.
                - segments (list[dict]): Per-segment details with timestamps.
                - duration_s (float): Total audio duration in seconds.
                - processing_time_s (float): Wall-clock inference time.
        """
        audio_path = str(audio_path)
        logger.info("Transcribing file: %s", audio_path)
        return self._run_transcription(audio_path)

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> dict:
        """
        Transcribe raw audio bytes (e.g. received over WebSocket).

        Writes bytes to a temporary file, transcribes, then cleans up.

        Args:
            audio_bytes: Raw audio file content.
            suffix: File extension hint for the temp file.

        Returns:
            Same dict structure as ``transcribe_file``.
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self._run_transcription(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_numpy(
        self, audio_array: np.ndarray, sample_rate: int = 16_000
    ) -> dict:
        """
        Transcribe a numpy audio array (float32, mono, 16 kHz expected).

        If the sample rate differs from 16 kHz the array is resampled
        internally by Whisper.

        Args:
            audio_array: 1-D float32 numpy array.
            sample_rate: Source sample rate (default 16000).

        Returns:
            Same dict structure as ``transcribe_file``.
        """
        import whisper

        # Whisper expects float32 numpy array at 16 kHz
        if sample_rate != 16_000:
            logger.info(
                "Resampling from %d Hz to 16000 Hz", sample_rate
            )
            audio_array = whisper.audio.resample(audio_array, sample_rate, 16_000)

        audio_array = audio_array.astype(np.float32)
        logger.info(
            "Transcribing numpy array: %.1f s of audio",
            len(audio_array) / 16_000,
        )
        return self._run_transcription(audio_array)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_transcription(self, audio_input) -> dict:
        """Core transcription logic shared by all public methods."""
        import whisper

        t0 = time.time()

        decode_options: dict = {
            "beam_size": self.beam_size,
            "word_timestamps": self.word_timestamps,
        }
        if self.language:
            decode_options["language"] = self.language

        result = whisper.transcribe(
            self.model,
            audio_input,
            **decode_options,
        )

        processing_time = time.time() - t0

        # Compute duration from segments
        duration = 0.0
        if result.get("segments"):
            duration = result["segments"][-1].get("end", 0.0)

        output = {
            "text": result.get("text", "").strip(),
            "language": result.get("language", self.language or "unknown"),
            "segments": _clean_segments(result.get("segments", [])),
            "duration_s": round(duration, 2),
            "processing_time_s": round(processing_time, 2),
        }

        logger.info(
            "Transcription complete — lang=%s, duration=%.1fs, "
            "processing=%.1fs, text_len=%d",
            output["language"],
            output["duration_s"],
            output["processing_time_s"],
            len(output["text"]),
        )
        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_segments(segments: list) -> list:
    """Normalise Whisper segment dicts to a consistent schema."""
    cleaned = []
    for seg in segments:
        entry = {
            "id": seg.get("id"),
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "text": seg.get("text", "").strip(),
        }
        # Include word-level timestamps if present
        if "words" in seg and seg["words"]:
            entry["words"] = [
                {
                    "word": w.get("word", "").strip(),
                    "start": round(w.get("start", 0.0), 2),
                    "end": round(w.get("end", 0.0), 2),
                    "probability": round(w.get("probability", 0.0), 3),
                }
                for w in seg["words"]
            ]
        cleaned.append(entry)
    return cleaned
