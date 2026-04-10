from flask import Flask, jsonify
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

# Import routes later to avoid circular imports
# from routes.audio import audio_bp
# from routes.session import session_bp
# from routes.patient import patient_bp
# from routes.report import report_bp

# from ws.stream_handler import register_ws_routes
# register_ws_routes(sock)

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "vaaksetu_backend"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
