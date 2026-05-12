# VaakSetu (वाक्-सेतु) — Voice Bridge

An AI-powered multilingual voice clinical intake assistant for ASHA workers and hospital nurses in Karnataka, India.

## ✨ Features

- **Voice-first intake** — Speak naturally, no typing required
- **Multilingual** — Code-mixed Kannada, Hindi, and English support
- **AI-powered extraction** — Automatic symptom detection, entity extraction, and follow-up question generation
- **Auto clinical records** — Structured reports generated from conversations
- **Text-to-speech** — Audio summaries in regional languages via Sarvam AI
- **Real-time processing** — WebSocket-based live transcription pipeline

## 🏗️ Architecture

| Layer | Technology |
|-------|-----------|
| **Frontend** | Vanilla JS, Jinja2 Templates, Custom CSS (Dark Theme) |
| **Backend** | Python Flask, Flask-SocketIO, SQLAlchemy, SQLite |
| **AI — ASR** | OpenAI Whisper, AI4Bharat IndicWhisper |
| **AI — LLM** | GPT-4o (OpenAI-compatible API) |
| **AI — TTS** | Sarvam AI Text-to-Speech |

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- Microphone access (for voice intake)

### Setup

```bash
# 1. Clone and enter the project
cd vaaksetu

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
# Edit .env with your API keys

# 5. Run the server
python app.py
```

Visit **http://localhost:5000** in your browser.

### Docker

```bash
docker-compose up
```

## 📁 Project Structure

```
vaaksetu/
├── app.py                 # Flask entry point
├── config.py              # Centralized configuration
├── requirements.txt       # Python dependencies
├── models/                # SQLAlchemy database models
├── routes/                # Flask Blueprints (API endpoints)
├── services/              # Business logic layer
├── ai_core/               # AI modules (ASR, LLM, Extraction, TTS)
├── templates/             # Jinja2 HTML templates
├── static/                # CSS, JavaScript, assets
│   ├── css/
│   └── js/
└── sample_data/           # Seed data for development
```

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | OpenAI / Groq API key |
| `LLM_BASE_URL` | LLM API base URL |
| `LLM_MODEL` | Model name (e.g. `gpt-4o`) |
| `SARVAM_API_KEY` | Sarvam AI TTS API key |
| `DATABASE_URL` | Database connection string |
| `FLASK_SECRET_KEY` | Flask session secret |
| `FLASK_DEBUG` | Enable debug mode (`true`/`false`) |
| `PORT` | Server port (default: `5000`) |
