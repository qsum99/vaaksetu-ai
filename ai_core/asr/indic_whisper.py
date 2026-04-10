"""
IndicWhisper ASR Engine — Kannada-native speech recognition.

Uses the AI4Bharat IndicWhisper model (fine-tuned Whisper for Indic languages)
via HuggingFace Transformers for high-accuracy Kannada transcription in
medical conversations.
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
# Lazy-loaded pipeline cache
# ---------------------------------------------------------------------------
_pipeline_cache: dict = {}

# Default HuggingFace model for Kannada ASR
DEFAULT_MODEL_ID = "ai4bharat/indicwhisper-hindi-large-v2"
# For Kannada specifically, you may swap to a Kannada-specific checkpoint
KANNADA_MODEL_ID = "ai4bharat/indicwhisper-large-v2"


def _get_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    torch_dtype=None,
):
    """Load and cache a HuggingFace ASR pipeline for IndicWhisper."""
    import torch
    from transformers import pipeline as hf_pipeline

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch_dtype is None:
        torch_dtype = torch.float16 if "cuda" in str(device) else torch.float32

    cache_key = f"{model_id}@{device}"
    if cache_key not in _pipeline_cache:
        logger.info("Loading IndicWhisper pipeline '%s' on %s …", model_id, device)
        t0 = time.time()
        _pipeline_cache[cache_key] = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            batch_size=8,
        )
        logger.info("IndicWhisper pipeline loaded in %.1f s", time.time() - t0)

    return _pipeline_cache[cache_key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class IndicWhisperASR:
    """
    IndicWhisper-based ASR for Kannada (and other Indic languages).

    Leverages AI4Bharat's fine-tuned Whisper models via HuggingFace
    Transformers for accurate transcription of Kannada medical dialogue,
    including code-mixed Kannada-English speech.

    Usage::

        asr = IndicWhisperASR()
        result = asr.transcribe_file("patient_kannada.wav")
        print(result["text"])
    """

    SUPPORTED_LANGUAGES = {"kn", "hi", "ta", "te", "ml", "bn", "gu", "mr"}

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        language: str = "kn",
        return_timestamps: bool = True,
    ):
        """
        Args:
            model_id: HuggingFace model identifier for IndicWhisper.
            device: "cuda:0" or "cpu". Auto-detected if None.
            language: Target language code (ISO 639-1). Default: "kn" (Kannada).
            return_timestamps: Whether to return word/chunk timestamps.
        """
        import torch

        self.model_id = model_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.return_timestamps = return_timestamps
        self._pipe = None  # lazy load

    @property
    def pipe(self):
        """Lazy-load the HuggingFace ASR pipeline."""
        if self._pipe is None:
            self._pipe = _get_pipeline(self.model_id, self.device)
        return self._pipe

    # ------------------------------------------------------------------
    # Transcription methods
    # ------------------------------------------------------------------

    def transcribe_file(self, audio_path: Union[str, Path]) -> dict:
        """
        Transcribe an audio file from disk.

        Args:
            audio_path: Path to WAV / MP3 / FLAC / OGG file.

        Returns:
            dict with keys: text, language, segments, duration_s,
            processing_time_s.
        """
        audio_path = str(audio_path)
        logger.info("IndicWhisper transcribing file: %s", audio_path)
        return self._run_transcription(audio_path)

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> dict:
        """
        Transcribe raw audio bytes (e.g. from WebSocket).

        Args:
            audio_bytes: Raw audio file content.
            suffix: File extension for temp file.

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
        Transcribe a numpy audio array.

        Args:
            audio_array: 1-D float32 numpy array.
            sample_rate: Source sample rate (default 16000 Hz).

        Returns:
            Same dict structure as ``transcribe_file``.
        """
        logger.info(
            "IndicWhisper transcribing numpy array: %.1f s of audio",
            len(audio_array) / sample_rate,
        )
        # HuggingFace pipeline accepts dict with raw array + sampling_rate
        audio_input = {
            "raw": audio_array.astype(np.float32),
            "sampling_rate": sample_rate,
        }
        return self._run_transcription(audio_input)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_transcription(self, audio_input) -> dict:
        """Core transcription via HuggingFace pipeline."""
        t0 = time.time()

        generate_kwargs = {
            "language": self.language,
            "task": "transcribe",
        }

        result = self.pipe(
            audio_input,
            return_timestamps=self.return_timestamps,
            generate_kwargs=generate_kwargs,
        )

        processing_time = time.time() - t0

        # Parse output — HF pipeline returns {"text": ..., "chunks": [...]}
        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])

        segments = _parse_chunks(chunks)

        # Estimate duration from last chunk
        duration = 0.0
        if segments:
            duration = segments[-1].get("end", 0.0)

        output = {
            "text": text,
            "language": self.language,
            "segments": segments,
            "duration_s": round(duration, 2),
            "processing_time_s": round(processing_time, 2),
            "model": self.model_id,
        }

        logger.info(
            "IndicWhisper transcription complete — lang=%s, duration=%.1fs, "
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

def _parse_chunks(chunks: list) -> list:
    """Convert HuggingFace pipeline chunks to our standard segment format."""
    segments = []
    for i, chunk in enumerate(chunks):
        ts = chunk.get("timestamp", (0.0, 0.0))
        start = ts[0] if ts[0] is not None else 0.0
        end = ts[1] if ts[1] is not None else start
        segments.append({
            "id": i,
            "start": round(start, 2),
            "end": round(end, 2),
            "text": chunk.get("text", "").strip(),
        })
    return segments
