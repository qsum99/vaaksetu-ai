"""
Audio REST Routes — File upload transcription endpoints.

Provides REST API endpoints for audio file upload and transcription
as an alternative to WebSocket streaming.
"""

import logging
import os
import tempfile
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

audio_bp = Blueprint("audio", __name__, url_prefix="/api/audio")

# ---------------------------------------------------------------------------
# Lazy singleton for the language detector
# ---------------------------------------------------------------------------
_detector = None


def _get_detector():
    """Get or create the shared LanguageDetector instance."""
    global _detector
    if _detector is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from ai_core.asr import LanguageDetector
        _detector = LanguageDetector()
    return _detector


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@audio_bp.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """
    Transcribe an uploaded audio file.

    Request:
        - Content-Type: multipart/form-data
        - Field "audio": audio file (WAV, MP3, FLAC, OGG, M4A)
        - Field "language" (optional): ISO 639-1 code or "auto"

    Response:
        {
            "text": "transcribed text",
            "language": "en",
            "engine": "whisper",
            "segments": [...],
            "duration_s": 12.5,
            "processing_time_s": 3.2
        }
    """
    # Validate file upload
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Use field 'audio'."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Get optional language parameter
    language = request.form.get("language", "auto")
    force_lang = None if language == "auto" else language

    # Determine file extension
    ext = Path(audio_file.filename).suffix or ".wav"
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    if ext.lower() not in allowed_extensions:
        return jsonify({
            "error": f"Unsupported format '{ext}'. Allowed: {allowed_extensions}"
        }), 400

    # Save to temp file and transcribe
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            audio_file.save(tmp)
            tmp_path = tmp.name

        logger.info(
            "Transcribing uploaded file: %s (%d bytes, lang=%s)",
            audio_file.filename,
            os.path.getsize(tmp_path),
            language,
        )

        detector = _get_detector()
        result = detector.detect_and_transcribe(
            audio_path=tmp_path,
            force_language=force_lang,
        )

        return jsonify({
            "text": result["text"],
            "language": result["language"],
            "engine": result.get("engine", "whisper"),
            "segments": result.get("segments", []),
            "duration_s": result.get("duration_s", 0),
            "processing_time_s": result.get("processing_time_s", 0),
            "detection_confidence": result.get("detection_confidence", 0),
        }), 200

    except Exception as e:
        logger.exception("Transcription error")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@audio_bp.route("/detect-language", methods=["POST"])
def detect_language():
    """
    Detect the spoken language of an uploaded audio file.

    Request:
        - Content-Type: multipart/form-data
        - Field "audio": audio file

    Response:
        {
            "language": "kn",
            "confidence": 0.92,
            "probabilities": {"kn": 0.92, "hi": 0.05, ...},
            "detection_time_s": 0.8
        }
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Use field 'audio'."}), 400

    audio_file = request.files["audio"]
    ext = Path(audio_file.filename).suffix or ".wav"

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            audio_file.save(tmp)
            tmp_path = tmp.name

        detector = _get_detector()
        result = detector.detect_language(tmp_path)

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Language detection error")
        return jsonify({"error": str(e)}), 500

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@audio_bp.route("/languages", methods=["GET"])
def supported_languages():
    """Return all supported languages grouped by ASR engine."""
    detector = _get_detector()
    return jsonify(detector.get_supported_languages()), 200
