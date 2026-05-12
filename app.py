"""
VaakSetu — Flask Application Entry Point
Serves API, WebSocket (SocketIO), templates, and static files.
"""
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO

from config import Config
from models.database import init_db, seed_sample_data, close_db

# ── Create Flask app ──────────────────────────────────────
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# ── CORS ──────────────────────────────────────────────────
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── SocketIO ──────────────────────────────────────────────
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Register Blueprints ──────────────────────────────────
from routes.patient_routes import patient_bp
from routes.report_routes import report_bp
from routes.session_routes import session_bp
from routes.audio_routes import audio_bp
from routes.tts_routes import tts_bp

app.register_blueprint(patient_bp)
app.register_blueprint(report_bp)
app.register_blueprint(session_bp)
app.register_blueprint(audio_bp)
app.register_blueprint(tts_bp)

# ── Register SocketIO events ─────────────────────────────
from routes.ws_routes import register_socketio_events
register_socketio_events(socketio)

# ── DB teardown ───────────────────────────────────────────
app.teardown_appcontext(close_db)


# ── Page Routes (serve Jinja2 templates) ──────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/intake")
def intake():
    return render_template("intake.html")


@app.route("/report")
def report():
    return render_template("report.html")


@app.route("/history")
def history():
    return render_template("history.html")


# ── Health check ──────────────────────────────────────────
@app.route("/api/health")
def health():
    return {"status": "ok", "service": "VaakSetu"}


# ── Startup ───────────────────────────────────────────────
with app.app_context():
    init_db()
    seed_sample_data()


if __name__ == "__main__":
    socketio.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        allow_unsafe_werkzeug=True,
    )
