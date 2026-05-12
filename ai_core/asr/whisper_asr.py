import whisper
import tempfile
import os

# Load model lazily to avoid blocking startup if not immediately needed
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading Whisper Large-v3 model...")
        _model = whisper.load_model("large-v3")
    return _model

def transcribe(audio_bytes: bytes, lang: str = None) -> dict:
    """
    Transcribes audio bytes using Whisper.
    Returns: { "text": "...", "language": "en|hi|...", "segments": [...] }
    """
    model = get_model()
    
    # Needs to be saved to a temp file because whisper expects a file path
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        if lang:
            result = model.transcribe(tmp_path, language=lang)
        else:
            result = model.transcribe(tmp_path)
            
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "segments": result.get("segments", [])
        }
    finally:
        os.remove(tmp_path)
