import re

def detect(text: str) -> str:
    """
    Detects the dominant language based on character ranges.
    Returns: "hi" for Hindi, "en" for English/Other
    
    Devanagari (Hindi) unicode range: \u0900-\u097F
    """
    hindi_chars = re.findall(r'[\u0900-\u097F]', text)
    
    total_len = len(text.strip())
    if total_len == 0:
        return "en"
        
    hi_ratio = len(hindi_chars) / total_len
    
    if hi_ratio > 0.3:
        return "hi"
        
    return "en"
