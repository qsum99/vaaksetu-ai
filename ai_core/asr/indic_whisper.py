import tempfile
import os
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# Lazy loading of model
_pipe = None

def get_pipeline():
    global _pipe
    if _pipe is None:
        if pipeline is None:
            raise ImportError("transformers is not installed. indic_whisper requires it.")
        print("Loading indicWhisper model...")
        # Fallback if indicWhisper too heavy or not available in demo
        try:
            _pipe = pipeline("automatic-speech-recognition", model="vasista22/whisper-kannada-base")
        except Exception as e:
            print(f"Failed to load indic whisper from vasista22: {e}")
            _pipe = None
    return _pipe

def transcribe(audio_bytes: bytes) -> str:
    pipe = get_pipeline()
    if not pipe:
        from ai_core.asr.whisper_asr import transcribe as fallback_transcribe
        # Fallback
        res = fallback_transcribe(audio_bytes, lang="kn")
        return res["text"]

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        result = pipe(tmp_path)
        return result.get("text", "").strip()
    finally:
        os.remove(tmp_path)
