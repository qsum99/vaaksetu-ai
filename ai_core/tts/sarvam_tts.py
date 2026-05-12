import requests
import os

def speak(text: str, lang: str = "kn-IN") -> bytes:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("Warning: SARVAM_API_KEY not defined. Returning dummy TTS bytes.")
        return b"DUMMY_AUDIO_BYTES"
        
    try:
        response = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"API-Subscription-Key": api_key},
            json={
                "inputs": [text], 
                "target_language_code": lang,
                "speaker": "meera", 
                "model": "bulbul:v1"
            },
            timeout=10
        )
        response.raise_for_status()
        
        # sarvam returns {"audios": ["base64_string"]} typically, 
        # or audio bytes directly depending on version. 
        # The user's prompt assumes it returns audio bytes in response.content.
        
        # We will parse base64 if it's JSON, else bytes
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            import base64
            data = response.json()
            return base64.b64decode(data.get("audios", [""])[0])
        else:
            return response.content
            
    except Exception as e:
        print(f"TTS Error: {e}")
        return b""
