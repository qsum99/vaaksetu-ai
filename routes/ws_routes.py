"""WebSocket routes — Flask-SocketIO real-time audio streaming pipeline."""
import json
from flask_socketio import emit
from routes.session_routes import active_sessions, _new_session


def register_socketio_events(socketio):
    """Register all SocketIO event handlers on the given socketio instance."""

    @socketio.on("connect")
    def handle_connect():
        print("[WS] Client connected")
        emit("status", {"type": "status", "message": "Connected to VaakSetu"})

    @socketio.on("disconnect")
    def handle_disconnect():
        print("[WS] Client disconnected")

    @socketio.on("start_session")
    def handle_start_session(data):
        session_id = data.get("session_id", "default")
        if session_id not in active_sessions:
            active_sessions[session_id] = _new_session(session_id)
        emit("session_started", {"session_id": session_id})
        print(f"[WS] Session started: {session_id}")

    @socketio.on("audio_chunk")
    def handle_audio_chunk(data):
        """
        Receives audio chunk bytes + session_id.
        Pipeline: ASR → Language Detection → LLM → Field extraction → emit results.
        """
        session_id = data.get("session_id", "default")
        audio_bytes = data.get("audio")

        if not audio_bytes:
            return

        if session_id not in active_sessions:
            active_sessions[session_id] = _new_session(session_id)

        session = active_sessions[session_id]

        try:
            # ── 1. ASR Transcription ──
            from ai_core.asr.whisper_asr import transcribe as whisper_transcribe
            asr_result = whisper_transcribe(audio_bytes)
            transcript_text = asr_result.get("text", "")
            detected_lang_code = asr_result.get("language", "en")

            # ── 2. Language Detection (text-based refinement) ──
            from ai_core.asr.lang_detector import detect as detect_lang
            lang = detect_lang(transcript_text) if transcript_text else detected_lang_code

        except Exception as e:
            print(f"[WS] ASR error: {e}")
            return

        if not transcript_text.strip():
            return

        session["transcript"] += transcript_text + " "
        session["detected_languages"].append(lang)

        # ── 3. Emit transcript ──
        emit("transcript", {"type": "transcript", "text": transcript_text, "lang": lang})

        try:
            # ── 4. LLM conversation manager ──
            from ai_core.llm.conversation_manager import get_response
            llm_res = get_response(session["context"], transcript_text)

            # ── 5. Emit follow-up question ──
            followup = llm_res.get("followup_question", "")
            if followup:
                session["context"].append({"role": "assistant", "content": followup})
                emit("followup", {"type": "followup", "question": followup})

            # ── 6. Emit extracted fields ──
            extracted = llm_res.get("extracted_fields", {})
            for key, val in extracted.items():
                if val and key != "symptoms":
                    session["extracted_fields"][key] = val
                    emit("field", {"type": "field", "field": key, "value": str(val)})

            # ── 7. Emit detected symptoms ──
            new_symptoms = llm_res.get("detected_symptoms", [])
            for sym in new_symptoms:
                if sym not in session["detected_symptoms"]:
                    session["detected_symptoms"].append(sym)
                    emit("symptom", {"type": "symptom", "value": sym})

        except Exception as e:
            print(f"[WS] LLM pipeline error: {e}")
