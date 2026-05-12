"""Audio routes — audio file transcription endpoint."""
from flask import Blueprint, request, jsonify

audio_bp = Blueprint("audio", __name__, url_prefix="/api/audio")


@audio_bp.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    filename = audio_file.filename or ""

    if not filename.endswith((".webm", ".wav", ".ogg", ".mp3")):
        return jsonify({"error": "Invalid audio format. Accepted: webm, wav, ogg, mp3"}), 400

    audio_bytes = audio_file.read()

    # TODO: Wire real ASR when models are loaded
    # from ai_core.asr.whisper_asr import transcribe
    # result = transcribe(audio_bytes)
    # return jsonify(result)

    return jsonify({"text": "dummy transcription", "language": "en"})
