import os
import sys

# Ensure src and root are in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
SAVED_MODEL_DIR = os.path.join(ROOT_DIR, "saved_model")

for d in [ROOT_DIR, SRC_DIR, SAVED_MODEL_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from src.app import build_ui, THEME, CUSTOM_CSS

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    server_name = os.environ.get("HOST", "0.0.0.0")
    print(f"[SafeDerm Triage Lite] Starting server on http://{server_name}:{port}...", flush=True)
    build_ui().launch(server_name=server_name, server_port=port, share=False, theme=THEME, css=CUSTOM_CSS)
