import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr
import uvicorn

# Ensure src, saved_model, and root are in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
SAVED_MODEL_DIR = os.path.join(ROOT_DIR, "saved_model")

for d in [ROOT_DIR, SRC_DIR, SAVED_MODEL_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from src.app import build_ui, get_backend

# 1. Create FastAPI application to handle PWA and upload progress endpoints cleanly
app = FastAPI(title="SafeDerm Triage Lite")

@app.get("/manifest.json")
def get_manifest():
    """Returns valid PWA manifest to prevent 404 in Chrome/Edge."""
    return JSONResponse({
        "name": "SafeDerm Triage Lite",
        "short_name": "SafeDerm",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#f8fafc",
        "theme_color": "#0B6E75"
    })

@app.get("/gradio_api/upload_progress")
@app.get("/upload_progress")
def get_upload_progress(upload_id: str = None):
    """Silently handles Gradio client upload progress polling to prevent 404 stream errors."""
    return JSONResponse({"status": "done", "progress": 100})

# 2. Pre-load model weights into memory at startup
print("[SafeDerm Triage Lite] Pre-loading model into memory...", flush=True)
try:
    get_backend()
    print("[SafeDerm Triage Lite] Model successfully loaded at startup!", flush=True)
except Exception as e:
    print(f"[SafeDerm Triage Lite WARNING] Startup model preload failed: {e}", flush=True)

# 3. Mount Gradio UI at root path
demo = build_ui()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    server_name = os.environ.get("HOST", "0.0.0.0")
    print(f"[SafeDerm Triage Lite] Starting Uvicorn server on http://{server_name}:{port}...", flush=True)
    uvicorn.run(app, host=server_name, port=port, log_level="info")
