from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response

# Using our tts module
from ai_core.tts import sarvam_tts

router = APIRouter(tags=["TTS"])

class TTSRequest(BaseModel):
    text: str
    lang: str = "kn-IN"

@router.post("/speak")
async def tts_speak(req: TTSRequest):
    try:
        audio_bytes = sarvam_tts.speak(req.text, req.lang)
        if not audio_bytes:
            raise ValueError("Empty audio returned")
            
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
