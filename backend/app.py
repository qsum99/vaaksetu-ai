from flask import Flask, jsonify
from flask_sock import Sock
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend cross-origin requests
sock = Sock(app)

# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------
from db import init_db
init_db()

# ---------------------------------------------------------------------------
# Register REST route blueprints
# ---------------------------------------------------------------------------
from routes.audio import audio_bp
from routes.session import session_bp
from routes.patient import patient_bp
from routes.report import report_bp

app.register_blueprint(audio_bp)
app.register_blueprint(session_bp)
app.register_blueprint(patient_bp)
app.register_blueprint(report_bp)

# ---------------------------------------------------------------------------
# Register WebSocket routes
# ---------------------------------------------------------------------------
from ws.stream_handler import register_ws_routes
register_ws_routes(sock)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint with database connectivity status."""
    from db import engine
    try:
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    return jsonify({
        "status": "healthy",
        "service": "vaaksetu_backend",
        "database": db_status,
    }), 200


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get platform statistics."""
    from db import get_db
    from models import Patient, ConsultationSession, ClinicalRecord

    db = get_db()
    try:
        return jsonify({
            "total_patients": db.query(Patient).count(),
            "total_sessions": db.query(ConsultationSession).count(),
            "active_sessions": db.query(ConsultationSession).filter(
                ConsultationSession.status == "active"
            ).count(),
            "total_clinical_records": db.query(ClinicalRecord).count(),
            "unreviewed_records": db.query(ClinicalRecord).filter(
                ClinicalRecord.reviewed_by_doctor == False
            ).count(),
        }), 200
    finally:
        db.close()


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    )
    app.run(host="0.0.0.0", port=5000, debug=True)
