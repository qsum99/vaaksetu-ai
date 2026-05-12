from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict
# We will integrate the ASR later
# from ai_core.asr.lang_detector import detect
# from ai_core.asr.whisper_asr import transcribe as whisper_transcribe
# from ai_core.asr.indic_whisper import transcribe as indic_transcribe

router = APIRouter(tags=["Audio"])

@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> Dict[str, str]:
    if not audio.filename.endswith(('.webm', '.wav', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid audio format")
    
    audio_bytes = await audio.read()
    # Mock ASR behavior for now until Step 4
    return {"text": "dummy transcription", "language": "en"}
