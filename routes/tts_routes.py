"""TTS routes — text-to-speech synthesis."""
from flask import Blueprint, request, jsonify, Response

tts_bp = Blueprint("tts", __name__, url_prefix="/api/tts")


@tts_bp.route("/speak", methods=["POST"])
def tts_speak():
    data = request.get_json()
    if not data or not data.get("text"):
        return jsonify({"error": "Text is required"}), 400

    text = data["text"]
    lang = data.get("lang", "kn-IN")

    try:
        from ai_core.tts.sarvam_tts import speak
        audio_bytes = speak(text, lang)

        if not audio_bytes:
            return jsonify({"error": "Empty audio returned from TTS"}), 500

        return Response(audio_bytes, mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500
