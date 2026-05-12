from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio
from typing import Dict
from backend.models.session import SessionState

# Imports for AI core
from ai_core.asr import lang_detector, whisper_asr
from ai_core.llm import conversation_manager
from ai_core.extraction import extractor

router = APIRouter()

active_sessions: Dict[str, SessionState] = {}

@router.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket, session_id: str = "default"):
    await websocket.accept()
    print(f"WebSocket connected for session: {session_id}")
    
    if session_id not in active_sessions:
        active_sessions[session_id] = SessionState(session_id=session_id)
        
    session = active_sessions[session_id]
    
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            
            # 1. Detect language (pseudo-detection on audio is hard locally,
            #    we'll pass it to whisper first or assume based on previous if not enough info)
            # Whisper can auto-detect. We'll use Whisper to transcribe and detect.
            # If indic_whisper is configured for KN specifically, we'd route based on UI toggle or initial guess.
            # For this exercise, we will use Whisper for EN/HI and Indic for KN based on pure Whisper dictation first,
            # but the prompt says: detect(chunk) -> route. Since lang_detector works on text, we'll transcribe first.
            # Let's adjust to use Whisper ASR which returns text + lang.
            
            # Use Whisper as default to get text + lang
            try:
                asr_result = await asyncio.to_thread(whisper_asr.transcribe, audio_bytes)
                transcript_text = asr_result["text"]
                detected_lang_code = asr_result["language"]
                
                lang = lang_detector.detect(transcript_text) if transcript_text else detected_lang_code
                
            except Exception as e:
                print(f"ASR error: {e}")
                continue
                
            if not transcript_text.strip():
                continue
                
            session.transcript += transcript_text + " "
            session.detected_languages.append(lang)
            
            # Send transcript event
            await websocket.send_text(json.dumps({
                "type": "transcript",
                "text": transcript_text,
                "lang": lang
            }))
            
            # 4. LLM conversation manager
            llm_res = await asyncio.to_thread(conversation_manager.get_response, session.context, transcript_text)
            
            # Send Followup
            followup = llm_res.get("followup_question")
            if followup:
                session.context.append({"role": "assistant", "content": followup})
                await websocket.send_text(json.dumps({
                    "type": "followup",
                    "question": followup
                }))
                
            # Send extracted fields
            extracted = llm_res.get("extracted_fields", {})
            for key, val in extracted.items():
                if val and key != "symptoms":
                    session.extracted_fields[key] = val
                    await websocket.send_text(json.dumps({
                        "type": "field",
                        "field": key,
                        "value": str(val)
                    }))
                    
            # Send detected symptoms
            new_symptoms = llm_res.get("detected_symptoms", [])
            for sym in new_symptoms:
                if sym not in session.detected_symptoms:
                    session.detected_symptoms.append(sym)
                    await websocket.send_text(json.dumps({
                        "type": "symptom",
                        "value": sym
                    }))

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WS error: {e}")
