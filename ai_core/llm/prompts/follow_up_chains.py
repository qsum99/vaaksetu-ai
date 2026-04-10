"""
Follow-up Prompt Templates — Symptom-specific question chains.

These templates guide the LLM to ask targeted follow-up questions
based on the patient's chief complaint and reported symptoms.
"""

# ---------------------------------------------------------------------------
# Symptom-specific follow-up chains
# ---------------------------------------------------------------------------

FOLLOW_UP_CHAINS = {
    "chest_pain": {
        "name": "Chest Pain Assessment",
        "questions": [
            "When did the chest pain start? Was it sudden or gradual?",
            "Can you describe the pain — is it sharp, dull, pressing, or burning?",
            "Does the pain spread anywhere — like your left arm, jaw, neck, or back?",
            "Does the pain get worse with physical activity, deep breathing, or eating?",
            "On a scale of 1 to 10, how would you rate the pain right now?",
            "Have you experienced any shortness of breath, sweating, or nausea along with the pain?",
            "Do you have any history of heart problems, high blood pressure, or diabetes?",
            "Are you currently taking any medications?",
        ],
        "red_flags": [
            "radiation to left arm or jaw",
            "pain with exertion",
            "associated shortness of breath",
            "cold sweats",
        ],
    },

    "fever": {
        "name": "Fever Assessment",
        "questions": [
            "When did the fever start? How many days has it been?",
            "Have you measured your temperature? If so, what was it?",
            "Is the fever constant, or does it come and go during the day?",
            "Are you having chills, shivering, or sweating?",
            "Do you have any body aches, headache, or fatigue along with the fever?",
            "Have you noticed any cough, cold, sore throat, or runny nose?",
            "Have you traveled anywhere recently or been in contact with anyone who was sick?",
            "Are you taking any medications for the fever?",
        ],
        "red_flags": [
            "temperature above 103°F (39.4°C)",
            "fever lasting more than 3 days",
            "confusion or altered consciousness",
            "rash with fever",
        ],
    },

    "headache": {
        "name": "Headache Assessment",
        "questions": [
            "When did the headache start?",
            "Where exactly is the pain — front, back, one side, or all over?",
            "What does the pain feel like — throbbing, pressing, sharp, or dull?",
            "On a scale of 1 to 10, how severe is it?",
            "Have you noticed any changes in your vision, nausea, or sensitivity to light?",
            "Does anything make it better or worse — like rest, medications, or certain positions?",
            "How often do you get headaches? Is this one different from your usual headaches?",
            "Do you have any history of migraines or head injuries?",
        ],
        "red_flags": [
            "worst headache of life (thunderclap)",
            "headache with fever and stiff neck",
            "sudden vision loss",
            "headache after head injury",
        ],
    },

    "abdominal_pain": {
        "name": "Abdominal Pain Assessment",
        "questions": [
            "Where in your abdomen is the pain — upper, lower, left, right, or center?",
            "When did the pain start? Was it sudden or gradual?",
            "What does the pain feel like — cramping, sharp, burning, or dull ache?",
            "Is the pain related to eating — does it get worse or better after meals?",
            "Have you had any changes in your bowel movements — diarrhea, constipation, or blood?",
            "Have you experienced any nausea, vomiting, or loss of appetite?",
            "On a scale of 1 to 10, how severe is the pain?",
            "Have you had any similar episodes before?",
        ],
        "red_flags": [
            "severe sudden pain",
            "blood in stool or vomit",
            "rigid or board-like abdomen",
            "pain with high fever",
        ],
    },

    "cough": {
        "name": "Cough Assessment",
        "questions": [
            "How long have you had the cough?",
            "Is it a dry cough or are you bringing up phlegm/mucus?",
            "If there's phlegm, what color is it — clear, yellow, green, or blood-tinged?",
            "Do you have any shortness of breath or wheezing?",
            "Is the cough worse at any particular time — morning, night, or with activity?",
            "Do you have any associated fever, body aches, or sore throat?",
            "Are you a smoker or exposed to dust/smoke regularly?",
            "Do you have any history of asthma, allergies, or lung problems?",
        ],
        "red_flags": [
            "coughing up blood",
            "severe shortness of breath",
            "cough lasting more than 3 weeks",
            "weight loss with chronic cough",
        ],
    },

    "joint_pain": {
        "name": "Joint Pain Assessment",
        "questions": [
            "Which joints are affected? Is it one joint or multiple?",
            "When did the joint pain start?",
            "Is there any swelling, redness, or warmth in the joint?",
            "Is the pain worse in the morning or after activity?",
            "Does the stiffness improve with movement or rest?",
            "Have you had any recent injuries or falls?",
            "Do you have any history of arthritis, gout, or autoimmune conditions?",
            "Are you taking any medications for the pain?",
        ],
        "red_flags": [
            "hot swollen joint with fever",
            "inability to bear weight",
            "joint deformity after injury",
        ],
    },

    "breathing_difficulty": {
        "name": "Breathing Difficulty Assessment",
        "questions": [
            "When did the breathing difficulty start?",
            "Is it worse at rest or with activity?",
            "Do you hear any wheezing or whistling sound when you breathe?",
            "Can you lie flat comfortably, or do you need to prop yourself up?",
            "Do you have any chest pain, cough, or fever with the breathlessness?",
            "Have you been exposed to any allergens, smoke, or chemicals recently?",
            "Do you have a history of asthma, COPD, or heart problems?",
            "Are you currently on any inhalers or breathing medications?",
        ],
        "red_flags": [
            "breathlessness at rest",
            "bluish discoloration of lips or fingers",
            "inability to speak in full sentences",
            "sudden onset with chest pain",
        ],
    },

    "skin_rash": {
        "name": "Skin Rash Assessment",
        "questions": [
            "When did the rash first appear?",
            "Where on your body did it start, and has it spread?",
            "What does it look like — red, raised, flat, blistered, or scaly?",
            "Is it itchy, painful, or burning?",
            "Have you started any new medications, foods, or products recently?",
            "Have you been in contact with anyone who has a similar rash?",
            "Do you have any fever, joint pain, or other symptoms along with the rash?",
            "Do you have any known allergies or skin conditions?",
        ],
        "red_flags": [
            "rapidly spreading rash with fever",
            "blistering or peeling skin",
            "rash with difficulty breathing",
            "petechial (non-blanching) rash",
        ],
    },

    "general": {
        "name": "General Assessment",
        "questions": [
            "Can you tell me more about what you're experiencing?",
            "When did this start?",
            "How severe would you say it is, on a scale of 1 to 10?",
            "Is it getting better, worse, or staying the same?",
            "Does anything make it better or worse?",
            "Do you have any other symptoms along with this?",
            "Do you have any existing medical conditions?",
            "Are you currently taking any medications or supplements?",
        ],
        "red_flags": [],
    },
}


# ---------------------------------------------------------------------------
# Follow-up selection prompt
# ---------------------------------------------------------------------------

SELECT_FOLLOW_UP_PROMPT = """\
Based on the conversation so far, determine the most appropriate next \
follow-up question to ask the patient.

Current conversation context:
{conversation_context}

Symptoms identified so far:
{symptoms_identified}

Questions already asked:
{questions_asked}

Rules:
1. Do NOT repeat a question that has already been asked or answered.
2. Prioritize questions about RED FLAGS first.
3. Ask about severity and duration early.
4. Move to medical history after symptom details are covered.
5. Ask ONE question at a time.
6. If all critical questions have been answered, indicate that a summary \
   can be generated.

Return your response as JSON:
{{
    "next_question": "the question to ask",
    "reason": "why this question is important",
    "category": "symptom_detail | red_flag | medical_history | lifestyle | summary_ready",
    "urgency": "high | medium | low"
}}\
"""
