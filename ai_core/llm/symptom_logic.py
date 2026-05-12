def get_followup_for_symptom(symptom: str) -> str:
    rules = {
        "chest pain": ["duration", "radiation", "breathing difficulty"],
        "fever": ["duration", "chills", "cough"],
        "vomiting": ["frequency", "blood in vomit", "last meal"],
        "headache": ["location", "intensity", "vision changes"],
        "stomach ache": ["exact location", "loose motion", "vomiting"]
    }
    
    symptom_lower = symptom.lower()
    for key, followups in rules.items():
        if key in symptom_lower:
            return f"Ask about {', '.join(followups)}."
            
    return "Ask for more details about the primary symptom."
