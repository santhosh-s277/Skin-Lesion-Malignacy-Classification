import os
import sys

# Ensure src and root are in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
SAVED_MODEL_DIR = os.path.join(ROOT_DIR, "saved_model")

for d in [ROOT_DIR, SRC_DIR, SAVED_MODEL_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from src.app import build_ui, THEME, CUSTOM_CSS, get_backend

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    server_name = os.environ.get("HOST", "0.0.0.0")
    print(f"[SafeDerm Triage Lite] Pre-loading model into memory...", flush=True)
    try:
        get_backend()
        print(f"[SafeDerm Triage Lite] Model successfully loaded at startup!", flush=True)
    except Exception as e:
        print(f"[SafeDerm Triage Lite WARNING] Startup model preload failed: {e}", flush=True)

    print(f"[SafeDerm Triage Lite] Starting server on http://{server_name}:{port}...", flush=True)
    demo = build_ui()
    demo.launch(server_name=server_name, server_port=port, share=False, theme=THEME, css=CUSTOM_CSS)

