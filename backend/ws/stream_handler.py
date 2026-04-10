"""
WebSocket Stream Handler — Real-time audio streaming for ASR.

Receives audio chunks from the browser via WebSocket, accumulates them,
and runs transcription using the LanguageDetector pipeline. Supports
both full-audio and chunked-streaming modes.
"""

import io
import json
import logging
import struct
import tempfile
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton for the language detector (avoids re-loading models)
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
# WebSocket route registration
# ---------------------------------------------------------------------------

def register_ws_routes(sock):
    """
    Register WebSocket endpoints on the Flask-Sock instance.

    Endpoints:
        /ws/transcribe  — Full audio blob transcription
        /ws/stream      — Chunked real-time streaming transcription
    """

    @sock.route("/ws/transcribe")
    def ws_transcribe(ws):
        """
        Full-audio transcription over WebSocket.

        Protocol:
            1. Client sends a JSON config message:
               {"type": "config", "language": "auto", "format": "wav"}
            2. Client sends raw audio bytes (single message).
            3. Server responds with JSON transcription result.
            4. Connection closes.
        """
        logger.info("WebSocket /ws/transcribe — client connected")

        try:
            # Step 1: Read config
            config_msg = ws.receive()
            config = _parse_config(config_msg)
            language = config.get("language", "auto")
            audio_format = config.get("format", "wav")

            logger.info(
                "Config received — language=%s, format=%s", language, audio_format
            )

            # Step 2: Receive audio data
            audio_data = ws.receive()
            if isinstance(audio_data, str):
                # If string, it might be base64
                import base64
                audio_data = base64.b64decode(audio_data)

            if not audio_data or len(audio_data) < 100:
                ws.send(json.dumps({
                    "type": "error",
                    "message": "Audio data too small or empty",
                }))
                return

            logger.info("Received %d bytes of audio data", len(audio_data))

            # Step 3: Transcribe
            ws.send(json.dumps({"type": "status", "message": "transcribing"}))

            detector = _get_detector()
            force_lang = None if language == "auto" else language

            result = detector.detect_and_transcribe(
                audio_bytes=audio_data,
                force_language=force_lang,
            )

            # Step 4: Send result
            response = {
                "type": "transcription",
                "text": result["text"],
                "language": result["language"],
                "engine": result.get("engine", "whisper"),
                "segments": result.get("segments", []),
                "duration_s": result.get("duration_s", 0),
                "processing_time_s": result.get("processing_time_s", 0),
            }
            ws.send(json.dumps(response))
            logger.info("Transcription sent — %d chars", len(result["text"]))

        except Exception as e:
            logger.exception("WebSocket transcription error")
            try:
                ws.send(json.dumps({
                    "type": "error",
                    "message": str(e),
                }))
            except Exception:
                pass

    @sock.route("/ws/stream")
    def ws_stream(ws):
        """
        Chunked real-time streaming transcription over WebSocket.

        Protocol:
            1. Client sends JSON config:
               {"type": "config", "language": "auto", "sample_rate": 16000}
            2. Client sends audio chunks (binary) continuously.
            3. Client sends JSON: {"type": "end"} to signal finish.
            4. Server responds with final transcription.

        Intermediate partial results are sent after every N seconds
        of accumulated audio.
        """
        logger.info("WebSocket /ws/stream — client connected")

        try:
            # Step 1: Config
            config_msg = ws.receive()
            config = _parse_config(config_msg)
            language = config.get("language", "auto")
            sample_rate = config.get("sample_rate", 16000)
            partial_interval_s = config.get("partial_interval_s", 5.0)

            logger.info(
                "Stream config — language=%s, sample_rate=%d, "
                "partial_interval=%.1fs",
                language, sample_rate, partial_interval_s,
            )

            ws.send(json.dumps({"type": "status", "message": "listening"}))

            # Step 2: Accumulate chunks
            audio_chunks = []
            total_bytes = 0
            chunk_count = 0
            last_partial_time = time.time()

            while True:
                message = ws.receive()

                # Check for end signal
                if isinstance(message, str):
                    try:
                        msg = json.loads(message)
                        if msg.get("type") == "end":
                            logger.info("Stream end signal received")
                            break
                    except json.JSONDecodeError:
                        pass
                    continue

                # Binary audio chunk
                audio_chunks.append(message)
                total_bytes += len(message)
                chunk_count += 1

                # Send partial transcription at intervals
                now = time.time()
                if (now - last_partial_time) >= partial_interval_s and total_bytes > 0:
                    partial_audio = b"".join(audio_chunks)
                    partial_wav = _raw_to_wav(partial_audio, sample_rate)

                    try:
                        detector = _get_detector()
                        force_lang = None if language == "auto" else language
                        partial_result = detector.detect_and_transcribe(
                            audio_bytes=partial_wav,
                            force_language=force_lang,
                        )
                        ws.send(json.dumps({
                            "type": "partial",
                            "text": partial_result["text"],
                            "language": partial_result["language"],
                        }))
                    except Exception as e:
                        logger.warning("Partial transcription failed: %s", e)

                    last_partial_time = now

            # Step 3: Final transcription
            if not audio_chunks:
                ws.send(json.dumps({
                    "type": "error",
                    "message": "No audio data received",
                }))
                return

            logger.info(
                "Stream complete — %d chunks, %d bytes total",
                chunk_count, total_bytes,
            )

            full_audio = b"".join(audio_chunks)
            full_wav = _raw_to_wav(full_audio, sample_rate)

            ws.send(json.dumps({"type": "status", "message": "transcribing"}))

            detector = _get_detector()
            force_lang = None if language == "auto" else language
            result = detector.detect_and_transcribe(
                audio_bytes=full_wav,
                force_language=force_lang,
            )

            response = {
                "type": "transcription",
                "text": result["text"],
                "language": result["language"],
                "engine": result.get("engine", "whisper"),
                "segments": result.get("segments", []),
                "duration_s": result.get("duration_s", 0),
                "processing_time_s": result.get("processing_time_s", 0),
            }
            ws.send(json.dumps(response))
            logger.info("Final transcription sent — %d chars", len(result["text"]))

        except Exception as e:
            logger.exception("WebSocket streaming error")
            try:
                ws.send(json.dumps({
                    "type": "error",
                    "message": str(e),
                }))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_config(message) -> dict:
    """Parse a config message (JSON string or dict)."""
    if isinstance(message, str):
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return {}
    elif isinstance(message, dict):
        return message
    return {}


def _raw_to_wav(
    raw_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """
    Wrap raw PCM bytes in a WAV container.

    Args:
        raw_bytes: Raw PCM audio bytes (16-bit signed LE).
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.
        sample_width: Bytes per sample (2 = 16-bit).

    Returns:
        Complete WAV file as bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    return buf.getvalue()
