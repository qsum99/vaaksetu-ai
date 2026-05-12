
import sys
import os
sys.path.append(os.getcwd())

print("Testing imports...")
try:
    from ai_core.asr import whisper_asr
    print("ASR Import OK")
    # from ai_core.tts import sarvam_tts
    # print("TTS Import OK")
    from ai_core.llm import conversation_manager
    print("LLM Import OK")
    from ai_core.extraction import extractor
    print("Extractor Import OK")
    
    print("\nTesting Model Lazy Loading (this might take a while if downloading)...")
    # model = whisper_asr.get_model()
    # print("Whisper Model Loaded OK")
    
except Exception as e:
    print(f"Import/Init Error: {e}")
    sys.exit(1)
