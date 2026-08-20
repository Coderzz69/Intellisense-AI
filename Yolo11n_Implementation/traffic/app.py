from __future__ import annotations

import mimetypes
import os
import socket
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from ui_backend import PROJECT_ROOT, get_job_state, submit_job

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

ALLOWED_MEDIA_ROOTS = [
    PROJECT_ROOT / "outputs",
]


def _find_available_port(host: str, preferred_port: int, attempts: int = 20) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"Could not find a free port in range {preferred_port}-{preferred_port + attempts - 1}."
    )


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/media/<path:relative_path>", methods=["GET"])
def media_file(relative_path: str):
    candidate = (PROJECT_ROOT / relative_path).resolve()
    if not candidate.is_file():
        abort(404)
    if not any(candidate.is_relative_to(root.resolve()) for root in ALLOWED_MEDIA_ROOTS):
        abort(403)
    mimetype, _ = mimetypes.guess_type(str(candidate))
    return send_file(candidate, conditional=True, mimetype=mimetype)


@app.route("/api/jobs", methods=["POST"])
def create_job():
    files = {
        "north_feed": request.files.get("north_feed"),
        "east_feed": request.files.get("east_feed"),
        "south_feed": request.files.get("south_feed"),
        "west_feed": request.files.get("west_feed"),
    }
    try:
        job_id = submit_job(files)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"jobId": job_id, "status": "queued"}), 202


@app.route("/api/jobs/<job_id>", methods=["GET"])
def job_state(job_id: str):
    state = get_job_state(job_id)
    if state is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(state)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    preferred_port = int(os.environ.get("PORT", "5001"))
    port = preferred_port if "PORT" in os.environ else _find_available_port(host, preferred_port)
    print(f"Traffic UI starting on http://127.0.0.1:{port}")
    app.run(host=host, port=port, debug=False)
