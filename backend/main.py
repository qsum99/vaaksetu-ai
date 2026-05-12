from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import audio, session, patient, report, tts
from backend.ws.stream_handler import router as ws_router
from backend.db.database import init_db, seed_sample_data

app = FastAPI(title="VaakSetu API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio.router, prefix="/audio")
app.include_router(session.router, prefix="/session")
app.include_router(patient.router, prefix="/patient")
app.include_router(report.router, prefix="/report")
app.include_router(tts.router, prefix="/tts")
app.include_router(ws_router)

@app.on_event("startup")
async def startup():
    init_db()
    seed_sample_data()
