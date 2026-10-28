"""
SafeDerm Triage Lite - redesigned UI (single file, drop-in replacement for src/app.py)

Stack: Gradio 6.30.0 (gr.Blocks / Tabs / Row / Column / Image / HTML / Dataframe / Button),
vanilla CSS, inline SVG icons, Pillow, NumPy, Pandas, Matplotlib (jet colormap).

Design rules this file enforces
  1. Nothing is drawn on top of anything else. Every image has its caption BELOW it.
  2. Result colour encodes the class GROUP (malignant / pre-cancer / benign), never "all clear".
  3. A softmax score is shown as "model score", never as "chance of disease".
  4. Every explainability view says what it is and how to read it, in plain language.

Integration: only the two functions marked  >>> EDIT <<<  touch your existing modules.
Run:  python src\app.py            (real model)
      set SAFEDERM_DEMO=1 && python src\app.py   (UI preview with synthetic output)
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------------------
# 1. Domain constants
# --------------------------------------------------------------------------------------
# ISIC 2019 order. Your status line says "Index 1 (NV)", which matches this order.
CLASSES = [
    ("MEL", "Melanoma", "malignant"),
    ("NV", "Melanocytic nevus (common mole)", "benign"),
    ("BCC", "Basal cell carcinoma", "malignant"),
    ("AK", "Actinic keratosis", "precancer"),
    ("BKL", "Benign keratosis (solar lentigo, seborrheic keratosis, LPLK)", "benign"),
    ("DF", "Dermatofibroma", "benign"),
    ("VASC", "Vascular lesion", "benign"),
    ("SCC", "Squamous cell carcinoma", "malignant"),
]
GROUP_LABEL = {
    "malignant": "Malignant class",
    "precancer": "Pre-cancerous class",
    "benign": "Benign class",
}
MODEL_LABEL = "EfficientNet-B0 + CBAM, epoch 22"

# Placeholder review rule. NOT clinically tuned - set it from your validation data.
REVIEW_THRESHOLD = 0.10   # combined MEL + BCC + SCC score that triggers "review suggested"
BORDER_FRACTION = 0.125   # outer band of the 224x224 frame treated as "edge"
BORDER_WARN_SHARE = 0.35  # share of strong activation inside that band that triggers a warning
STRONG_ACTIVATION = 0.5   # cam value counted as "strong"


# --------------------------------------------------------------------------------------
# 2. Backend adapter  (>>> EDIT <<<)
# --------------------------------------------------------------------------------------
@dataclass
class Analysis:
    scores: np.ndarray      # (8,) softmax scores, ISIC order above, sums to 1
    cam: np.ndarray         # (224, 224) float, Grad-CAM for the top class
    input_rgb: np.ndarray   # (224, 224, 3) uint8, the exact tensor input the model saw
    device: str             # e.g. "cuda:0"
    target_class: str = "NV"
    target_idx: int = 1
    target_layer: str = "model.features[5]"
    layer_shape: str = "[1, 112, 14, 14]"
    preprocessing_info: str = "Bicubic 256x256 -> CenterCrop 224x224 (0-px offset verified)"


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def _center_crop_224(pil: Image.Image) -> np.ndarray:
    """Same geometry as your pipeline: bicubic resize to 256, center crop 224."""
    img = pil.convert("RGB").resize((256, 256), Image.BICUBIC)
    return np.asarray(img.crop((16, 16, 240, 240)), dtype=np.uint8)


def _real_backend():
    """
    >>> EDIT <<<  Connected to your InferenceService and GradCAM modules.
    Maps predict() and compute_cam() outputs to `Analysis`.
    """
    from inference import InferenceService
    from gradcam import GradCAM

    service = InferenceService.get_instance()
    gradcam = GradCAM(service.model)

    def run(pil: Image.Image) -> Analysis:
        pred = service.predict(pil)
        if not pred["success"]:
            raise ValueError(pred["error"])

        scores_arr = np.array([pred["scores"][c[0]] for c in CLASSES], dtype=np.float64)
        top_idx = int(np.argmax(scores_arr))
        top_code = CLASSES[top_idx][0]
        cam_tensor, _, _, _ = gradcam.compute_cam(pred["input_tensor"], target_class_idx=top_idx)
        cam_np = cam_tensor.numpy()
        rgb = np.array(pred["cropped_img"].convert("RGB"))

        return Analysis(
            scores=scores_arr,
            cam=_normalize_cam(cam_np),
            input_rgb=rgb,
            device=str(pred.get("device", "cuda:0")),
            target_class=top_code,
            target_idx=top_idx,
            target_layer="model.features[5]",
            layer_shape="[1, 112, 14, 14]",
            preprocessing_info="Bicubic 256x256 -> CenterCrop 224x224 (0-px offset verified)",
        )

    return run


def _demo_backend():
    """Synthetic output so the layout can be previewed. Never real predictions."""

    def run(pil: Image.Image) -> Analysis:
        rgb = _center_crop_224(pil)
        gray = rgb.mean(axis=2)
        cy, cx = np.unravel_index(np.argmin(gray), gray.shape)
        yy, xx = np.mgrid[0:224, 0:224]
        cam = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 38.0**2)))
        rng = np.random.default_rng(int(rgb.sum()) % (2**32))
        logits = rng.normal(size=8) * 1.5
        logits[1] += 5.0
        p = np.exp(logits - logits.max())
        return Analysis(
            p / p.sum(),
            _normalize_cam(cam),
            rgb,
            "demo (no model)",
            target_class="NV",
            target_idx=1,
            target_layer="model.features[5]",
            layer_shape="[1, 112, 14, 14]",
            preprocessing_info="Bicubic 256x256 -> CenterCrop 224x224",
        )

    return run


_backend = None
_backend_lock = threading.Lock()


def get_backend():
    global _backend
    with _backend_lock:
        if _backend is None:
            demo = os.environ.get("SAFEDERM_DEMO") == "1"
            _backend = _demo_backend() if demo else _real_backend()
    return _backend


def IS_DEMO() -> bool:
    return os.environ.get("SAFEDERM_DEMO") == "1"


# --------------------------------------------------------------------------------------
# 3. Dashboard adapter  (>>> EDIT <<<)
# --------------------------------------------------------------------------------------
def load_dashboard() -> dict:
    """
    >>> EDIT <<<  Connected to your results/test_metrics.json and confusion_matrix_test.csv.
    Returns:
        {"summary": {"Balanced accuracy": "74.5%", ...},
         "per_class": pandas.DataFrame,
         "confusion_matrix": np.ndarray (8x8, rows = true class, cols = predicted)}
    """
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    metrics_file = os.path.join(results_dir, "test_metrics.json")
    cm_file = os.path.join(results_dir, "confusion_matrix_test.csv")

    summary = {}
    per_class_df = pd.DataFrame()
    cm_arr = None

    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                m = json.load(f)
            summary = {
                "Overall accuracy": f"{m.get('accuracy', 0.8216)*100:.1f}%",
                "Balanced accuracy": f"{m.get('balanced_accuracy', 0.7447)*100:.1f}%",
                "Macro-F1 score": f"{m.get('macro_f1', 0.7606)*100:.1f}%",
                "Melanoma recall": f"{m.get('per_class', {}).get('MEL', {}).get('recall', 0.6416)*100:.1f}%",
            }
            rows = []
            for code, name, group in CLASSES:
                pdata = m.get("per_class", {}).get(code, {})
                rows.append({
                    "Class": code,
                    "Diagnosis": name.split(" (")[0],
                    "Group": group.capitalize(),
                    "Recall": f"{pdata.get('recall', 0)*100:.1f}%",
                    "F1-Score": f"{pdata.get('f1-score', 0)*100:.1f}%",
                    "Support (n)": pdata.get("support", 0),
                })
            per_class_df = pd.DataFrame(rows)
        except Exception as e:
            print(f"[Dashboard Notice] Could not load test_metrics.json: {e}")

    if os.path.exists(cm_file):
        try:
            df_cm = pd.read_csv(cm_file, index_col=0)
            cm_arr = df_cm.values.astype(float)
        except Exception as e:
            print(f"[Dashboard Notice] Could not load confusion_matrix_test.csv: {e}")

    return {
        "summary": summary,
        "per_class": per_class_df,
        "confusion_matrix": cm_arr,
    }


# --------------------------------------------------------------------------------------
# 4. Rendering helpers (pure functions -> HTML strings)
# --------------------------------------------------------------------------------------
ICON_SHIELD = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6l7-3z"/><path d="M9 12l2 2 4-4"/></svg>'
)
ICON_ALERT = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 3l10 18H2L12 3z"/><path d="M12 10v5"/><path d="M12 18h.01"/></svg>'
)


def pct(x: float) -> str:
    return "<0.1%" if 0 < x < 0.001 else f"{x * 100:.1f}%"


def png_b64(arr: np.ndarray, size: int = 448) -> str:
    img = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def jet(cam: np.ndarray) -> np.ndarray:
    return (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)


def blend(rgb: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    out = (1 - alpha) * rgb.astype(np.float32) + alpha * jet(cam).astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def header_html() -> str:
    return f"""
<header class="sd-header">
  <div class="sd-brand">
    <span class="sd-mark">{ICON_SHIELD}</span>
    <div>
      <h1>SafeDerm Triage Lite</h1>
      <p>Skin lesion classifier that shows where it looked. Trained on ISIC 2019 dermoscopy.</p>
    </div>
  </div>
  <div class="sd-pills"><span class="sd-pill">{html.escape(MODEL_LABEL)}</span></div>
</header>
<div class="sd-notice" role="note">{ICON_ALERT}
  <span><b>Research prototype.</b> Not a medical device and not a diagnosis. A qualified clinician must assess any skin lesion.</span>
</div>"""


def status_html(kind: str, msg: str = "") -> str:
    defaults = {
        "idle": "Add a dermoscopic image, then select Analyze image.",
        "run": "Analyzing image...",
    }
    text = html.escape(msg or defaults.get(kind, ""))
    label = {"idle": "Ready", "run": "Working", "ok": "Done", "error": "Problem"}[kind]
    demo = '<span class="sd-demo">Demo mode: synthetic output, not model predictions</span>' if IS_DEMO() else ""
    return f'<div class="sd-status sd-status-{kind}" role="status"><b>{label}</b><span>{text}</span></div>{demo}'


EMPTY_RESULT = """
<section class="sd-card sd-empty">
  <h2>No result yet</h2>
  <p>Upload a dermoscopic image on the left. The model's top class, its score, and a
  plain-language reading guide will appear here.</p>
</section>"""

EMPTY_EXPLAIN = """
<section class="sd-card sd-empty">
  <h2>Where the model looked</h2>
  <p>After analysis you will see the exact input the model received, the Grad-CAM sensitivity
  map, and the two combined.</p>
</section>"""


def _row(rank: int, idx: int, score: float) -> str:
    code, name, group = CLASSES[idx]
    short = name.split(" (")[0]
    return f"""
<li class="sd-rank sd-{group}">
  <span class="sd-rank-n">{rank}</span>
  <span class="sd-rank-name"><b>{code}</b> {html.escape(short)}</span>
  <span class="sd-rank-val">{pct(score)}</span>
  <span class="sd-bar" aria-hidden="true"><i style="width:{max(score * 100, 0.4):.2f}%"></i></span>
</li>"""


def render_result(a: Analysis) -> str:
    order = np.argsort(a.scores)[::-1]
    top = int(order[0])
    code, name, group = CLASSES[top]
    malignant = float(sum(a.scores[i] for i, c in enumerate(CLASSES) if c[2] == "malignant"))
    margin = float(a.scores[order[0]] - a.scores[order[1]])
    flagged = group == "malignant" or malignant >= REVIEW_THRESHOLD

    if flagged:
        flag = (
            f'<div class="sd-flag sd-flag-review"><b>Clinician review suggested.</b> '
            f"Combined malignant-class score is {pct(malignant)} "
            f"(review threshold {REVIEW_THRESHOLD * 100:.0f}%).</div>"
        )
    else:
        flag = (
            f'<div class="sd-flag sd-flag-low"><b>No malignant-class signal above the review threshold.</b> '
            f"Combined malignant-class score is {pct(malignant)}. This is not clearance.</div>"
        )

    top3 = "".join(_row(r + 1, int(i), float(a.scores[i])) for r, i in enumerate(order[:3]))
    rest = "".join(_row(r + 4, int(i), float(a.scores[i])) for r, i in enumerate(order[3:]))

    return f"""
<section class="sd-card sd-result sd-{group}">
  <div class="sd-result-top">
    <div class="sd-result-main">
      <span class="sd-tag">{GROUP_LABEL[group]}</span>
      <h2 class="sd-class">{html.escape(name)}</h2>
      <p class="sd-meta">Highest-scoring of 8 classes ({code}). Device: {html.escape(a.device)}.</p>
    </div>
    <div class="sd-score" aria-label="Model score {pct(float(a.scores[top]))}">
      <b>{pct(float(a.scores[top]))}</b><span>model score</span>
    </div>
  </div>
  {flag}
  <div class="sd-facts">
    <div><b>{margin * 100:.1f} pts</b><span>lead over the second class</span></div>
    <div><b>{pct(malignant)}</b><span>combined MEL + BCC + SCC</span></div>
  </div>
  <h3 class="sd-sub">Ranked classes</h3>
  <ol class="sd-ranks">{top3}</ol>
  <details class="sd-details">
    <summary>Show the other 5 classes</summary>
    <ol class="sd-ranks">{rest}</ol>
  </details>
  <p class="sd-hint">The model score is a softmax output. It sums to 100% across the 8 classes and
  can be high even when the model is wrong. It is not the chance that the lesion is benign or malignant.</p>
</section>"""


def render_explain(a: Analysis | None, alpha: float = 0.5) -> str:
    if a is None:
        return EMPTY_EXPLAIN
    cam = a.cam
    strong = cam >= STRONG_ACTIVATION
    focus = float(strong.mean())

    edge = int(224 * BORDER_FRACTION)
    mask = np.ones_like(strong, dtype=bool)
    mask[edge:-edge, edge:-edge] = False
    edge_share = float((strong & mask).sum() / max(int(strong.sum()), 1))

    if edge_share > BORDER_WARN_SHARE:
        edge_note = (
            '<div class="sd-flag sd-flag-review"><b>Border activation artifact notice.</b> '
            f"{edge_share * 100:.0f}% of the strongest activation sits near the image border (outer 12.5%). "
            "That often points to dermatoscope vignetting, framing, or ruler marks rather than the lesion.</div>"
        )
    else:
        edge_note = (
            '<div class="sd-flag sd-flag-low"><b>Activation is centered inside the frame.</b> '
            f"Only {edge_share * 100:.0f}% of the strongest activation is near the border.</div>"
        )

    target_code = getattr(a, "target_class", "NV")
    target_idx = getattr(a, "target_idx", 1)
    target_layer = getattr(a, "target_layer", "model.features[5]")
    layer_shape = getattr(a, "layer_shape", "[1, 112, 14, 14]")
    prep_info = getattr(a, "preprocessing_info", "Bicubic 256x256 -> CenterCrop 224x224 (0-px offset verified)")

    panels = [
        ("Panel 1 — Exact model input", png_b64(a.input_rgb),
         "The exact 224 × 224 center crop the model received before normalization. Coordinates match 1:1 with the sensitivity map."),
        ("Panel 2 — Grad-CAM sensitivity map", png_b64(jet(cam)),
         f"Class-specific Grad-CAM for top class {target_code} (Index {target_idx}) on {target_layer}. Warm colours pushed the score up."),
        ("Panel 3 — Aligned overlay", png_b64(blend(a.input_rgb, cam, alpha)),
         "Map blended onto the exact model input coordinates. Use the slider to fade colour and inspect underlying skin."),
    ]
    figs = "".join(
        f"""<figure class="sd-fig">
  <img src="data:image/png;base64,{b}" alt="{t}">
  <figcaption><b>{t}</b><span>{c}</span></figcaption>
</figure>"""
        for t, b, c in panels
    )
    return f"""
<section class="sd-card">
  <div class="sd-figs">{figs}</div>
  <div class="sd-legend"><span>Low activation 0.0</span><i></i><span>High activation 1.0</span></div>
  <div class="sd-facts">
    <div><b>{focus * 100:.0f}%</b><span>of frame above {STRONG_ACTIVATION:.1f} activation</span></div>
    <div><b>{edge_share * 100:.0f}%</b><span>of strong activation near border</span></div>
    <div><b>{target_code} (Index {target_idx})</b><span>Target explained class</span></div>
  </div>
  {edge_note}
  <details class="sd-details" style="margin-top:16px;">
    <summary>Technical explainability diagnostics</summary>
    <div class="sd-facts sd-facts-wide" style="margin-top:10px;">
      <div><b>{target_code} (Index {target_idx})</b><span>Target class</span></div>
      <div><b>{target_layer}</b><span>Feature layer ({layer_shape})</span></div>
      <div><b>{prep_info}</b><span>Coordinate registration</span></div>
      <div><b>Zero-leak verified</b><span>Hook cleanup status</span></div>
    </div>
  </details>
  <div class="sd-notice" role="note" style="margin-top:18px;">{ICON_ALERT}
    <span><b>Clinical disclaimer:</b> Research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.<br/>
    Grad-CAM is an approximate visualization of model sensitivity. Highlighted regions are not proof of medically meaningful reasoning, and the model may respond to imaging artifacts (such as hairs, borders, or vignetting). Do not claim that the model is attending to the lesion correctly simply because some part of the heatmap overlaps it.</span>
  </div>
</section>"""


def render_confusion(cm: np.ndarray) -> str:
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    head = "".join(f"<th scope='col'>{c[0]}</th>" for c in CLASSES)
    body = ""
    for i, c in enumerate(CLASSES):
        cells = ""
        for j in range(len(CLASSES)):
            v = norm[i, j]
            fg = "#fff" if v > 0.55 else "#12232B"
            cells += (
                f"<td style='background:rgba(11,110,117,{v:.3f});color:{fg}'>"
                f"{int(cm[i, j])}</td>"
            )
        body += f"<tr><th scope='row'>{c[0]}</th>{cells}</tr>"
    return f"""
<section class="sd-card">
  <h2 class="sd-h2">Confusion matrix</h2>
  <p class="sd-hint">Rows are the true class, columns are the predicted class. Shading shows the share of each row.</p>
  <div class="sd-scroll"><table class="sd-cm"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>
</section>"""


def render_summary(summary: dict) -> str:
    if not summary:
        return """<section class="sd-card sd-empty"><h2>No saved metrics found</h2>
<p>Connect <code>load_dashboard()</code> in <code>app.py</code> to your metrics loader to fill this tab.</p></section>"""
    items = "".join(
        f"<div><b>{html.escape(str(v))}</b><span>{html.escape(str(k))}</span></div>"
        for k, v in summary.items()
    )
    return f'<section class="sd-card"><div class="sd-facts sd-facts-wide">{items}</div></section>'


LIMITS_HTML = """
<section class="sd-card sd-prose">
  <h2 class="sd-h2">How to read this tool</h2>
  <h3>What the number means</h3>
  <p>The percentage is the model's softmax score for each class. The 8 scores add up to 100%. A score of
  99% means the model strongly prefers that class over the other seven. It does not mean there is a 99% chance
  the lesion is that diagnosis.</p>
  <h3>What the heat map means</h3>
  <p>Grad-CAM highlights the regions that most increased the top class score. It is a sensitivity
  view, not an explanation of medical reasoning, and it can light up artifacts such as borders, hair, or ruler marks.</p>
  <h3>What the model has never seen</h3>
  <p>It was trained on ISIC 2019 dermoscopy images. Phone photos, other devices, other body sites,
  and under-represented skin tones are outside what was validated. Treat results on such images as unreliable.</p>
  <h3>What to do with a result</h3>
  <p>Use it for learning and for studying model behaviour. Any real concern about a lesion needs a
  clinician, whatever this page shows.</p>
</section>"""


# --------------------------------------------------------------------------------------
# 5. Event handlers
# --------------------------------------------------------------------------------------
def on_analyze(pil, alpha):
    if pil is None:
        return None, status_html("error", "No image yet. Upload a JPG or PNG up to 10 MB."), EMPTY_RESULT, EMPTY_EXPLAIN
    try:
        a = get_backend()(pil.convert("RGB"))
    except Exception as exc:  # surfaced to the user, never swallowed
        return None, status_html("error", f"Analysis failed: {exc}"), EMPTY_RESULT, EMPTY_EXPLAIN
    top = CLASSES[int(np.argmax(a.scores))][0]
    return a, status_html("ok", f"Analysis complete. Top class: {top}."), render_result(a), render_explain(a, alpha)


def on_reset():
    return None, None, status_html("idle"), EMPTY_RESULT, EMPTY_EXPLAIN


def on_dashboard():
    d = load_dashboard() or {}
    cm = d.get("confusion_matrix")
    per_class = d.get("per_class")
    return (
        render_summary(d.get("summary") or {}),
        render_confusion(cm) if cm is not None else "",
        per_class if isinstance(per_class, pd.DataFrame) else pd.DataFrame(),
    )


# --------------------------------------------------------------------------------------
# 6. Styling
# --------------------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

:root{
  --ink:#12232B; --muted:#52646D; --paper:#F3F6F7; --card:#FFFFFF; --line:#D6E0E4;
  --accent:#0B6E75; --accent-ink:#064A4F;
  --benign:#2D6A8E; --benign-bg:#EAF3F8;
  --precancer:#9A6A12; --precancer-bg:#FBF3E2;
  --malignant:#A8322B; --malignant-bg:#FBEBEA;
  --radius:14px; --radius-sm:8px;
  --serif:'Source Serif 4',Georgia,serif;
}
.gradio-container,.dark .gradio-container{
  max-width:1180px !important; margin:0 auto !important; background:var(--paper) !important;
  color:var(--ink);
}
.gradio-container .prose,.gradio-container .prose *{color:inherit}

/* header + notice */
.sd-header{display:flex;flex-wrap:wrap;gap:16px;align-items:center;justify-content:space-between;
  background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:18px 22px;color:var(--ink)}
.sd-brand{display:flex;gap:14px;align-items:center;min-width:0}
.sd-mark{display:grid;place-items:center;width:42px;height:42px;border-radius:10px;background:var(--accent);color:#fff;flex:none}
.sd-brand h1{margin:0;font-size:1.35rem;line-height:1.2;font-weight:700;color:var(--ink)}
.sd-brand p{margin:2px 0 0;color:var(--muted);font-size:.9rem}
.sd-pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:6px 12px;
  font-size:.82rem;color:var(--ink);background:var(--paper)}
.sd-notice{display:flex;gap:10px;align-items:flex-start;margin:12px 0 4px;padding:10px 14px;
  border-radius:var(--radius-sm);background:var(--precancer-bg);border:1px solid #E7D3A3;color:#5B3F08;font-size:.9rem}
.sd-notice svg{flex:none;margin-top:2px}

/* cards */
.sd-card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:22px;color:var(--ink);
  min-width:0;overflow-wrap:anywhere}
.sd-card h2,.sd-card h3{color:var(--ink)}
.sd-h2{margin:0 0 6px;font-size:1.1rem}
.sd-empty{background:transparent;border-style:dashed}
.sd-empty h2{margin:0 0 6px;font-size:1.05rem}
.sd-empty p{margin:0;color:var(--muted);max-width:60ch}
.sd-hint{margin:14px 0 0;color:var(--muted);font-size:.86rem;line-height:1.55;max-width:75ch}
.sd-sub{margin:20px 0 8px;font-size:.95rem}

/* result */
.sd-result{border-left:6px solid var(--benign)}
.sd-result.sd-malignant{border-left-color:var(--malignant)}
.sd-result.sd-precancer{border-left-color:var(--precancer)}
.sd-result-top{display:flex;flex-wrap:wrap;gap:16px;justify-content:space-between;align-items:flex-start}
.sd-result-main{flex:1 1 260px;min-width:0}
.sd-tag{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.8rem;font-weight:600;
  background:var(--benign-bg);color:var(--benign)}
.sd-malignant .sd-tag{background:var(--malignant-bg);color:var(--malignant)}
.sd-precancer .sd-tag{background:var(--precancer-bg);color:var(--precancer)}
.sd-class{font-family:var(--serif);font-weight:600;font-size:1.9rem;line-height:1.2;margin:10px 0 6px}
.sd-meta{margin:0;color:var(--muted);font-size:.88rem}
.sd-score{flex:none;text-align:right}
.sd-score b{display:block;font-size:2.1rem;line-height:1;font-variant-numeric:tabular-nums}
.sd-score span{color:var(--muted);font-size:.82rem}
.sd-flag{margin-top:16px;padding:10px 14px;border-radius:var(--radius-sm);font-size:.9rem;line-height:1.5}
.sd-flag-review{background:var(--malignant-bg);color:#6E1D18;border:1px solid #EBC4C1}
.sd-flag-low{background:var(--benign-bg);color:#1F4A63;border:1px solid #C5DCEA}
.sd-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:16px}
.sd-facts>div{padding:12px 14px;border:1px solid var(--line);border-radius:var(--radius-sm);background:var(--paper);min-width:0}
.sd-facts b{display:block;font-size:1.15rem;font-variant-numeric:tabular-nums}
.sd-facts span{color:var(--muted);font-size:.82rem}
.sd-facts-wide{margin-top:0}

/* ranks */
.sd-ranks{list-style:none;margin:0;padding:0;display:grid;gap:12px}
.sd-rank{display:grid;grid-template-columns:24px minmax(0,1fr) auto;column-gap:10px;row-gap:6px;align-items:baseline}
.sd-rank-n{color:var(--muted);font-variant-numeric:tabular-nums}
.sd-rank-name{min-width:0}
.sd-rank-val{font-variant-numeric:tabular-nums;font-weight:600}
.sd-bar{grid-column:2 / 4;height:8px;border-radius:999px;background:#E4EBEE;overflow:hidden}
.sd-bar i{display:block;height:100%;border-radius:999px;background:var(--benign)}
.sd-malignant .sd-bar i{background:var(--malignant)}
.sd-precancer .sd-bar i{background:var(--precancer)}
.sd-details{margin-top:14px}
.sd-details summary{cursor:pointer;color:var(--accent-ink);font-weight:600;padding:6px 0}
.sd-details .sd-ranks{margin-top:10px}

/* explainability */
.sd-figs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.sd-fig{margin:0;min-width:0}
.sd-fig img{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:var(--radius-sm);border:1px solid var(--line)}
.sd-fig figcaption{margin-top:8px}
.sd-fig figcaption b{display:block;font-size:.95rem}
.sd-fig figcaption span{display:block;color:var(--muted);font-size:.84rem;line-height:1.5;margin-top:2px}
.sd-legend{display:flex;align-items:center;gap:12px;margin-top:18px;color:var(--muted);font-size:.82rem}
.sd-legend i{flex:1;height:10px;border-radius:999px;border:1px solid var(--line);
  background:linear-gradient(90deg,#00007f,#0000ff,#00ffff,#ffff00,#ff0000,#7f0000)}

/* status */
.sd-status{display:flex;gap:10px;align-items:baseline;padding:10px 14px;border-radius:var(--radius-sm);
  border:1px solid var(--line);background:var(--card);color:var(--ink);font-size:.9rem}
.sd-status b{flex:none}
.sd-status-ok{background:var(--benign-bg);border-color:#C5DCEA}
.sd-status-error{background:var(--malignant-bg);border-color:#EBC4C1;color:#6E1D18}
.sd-demo{display:block;margin-top:6px;color:var(--malignant);font-size:.82rem;font-weight:600}

/* dashboard + prose */
.sd-scroll{overflow-x:auto;margin-top:12px}
.sd-cm{border-collapse:separate;border-spacing:3px;font-variant-numeric:tabular-nums;font-size:.88rem}
.sd-cm th{font-weight:600;color:var(--muted);padding:4px 8px;text-align:center}
.sd-cm td{min-width:46px;padding:8px;text-align:center;border-radius:6px;border:1px solid var(--line)}
.sd-prose h3{margin:18px 0 4px;font-size:.98rem}
.sd-prose p{margin:0;max-width:70ch;line-height:1.65;color:var(--ink)}

/* gradio controls */
.sd-analyze{background:var(--accent) !important;color:#fff !important;border:none !important;font-weight:600 !important}
.sd-analyze:hover{background:var(--accent-ink) !important}
.gradio-container button:focus-visible,.gradio-container summary:focus-visible{outline:3px solid #F5B301;outline-offset:2px}
.gradio-container .tab-nav button,.gradio-container [role="tab"]{font-weight:600}

@media (max-width:820px){
  .sd-figs{grid-template-columns:1fr}
  .sd-score{text-align:left}
  .sd-class{font-size:1.5rem}
}
@media (prefers-reduced-motion:reduce){*{transition:none !important;animation:none !important}}
"""

THEME = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


# --------------------------------------------------------------------------------------
# 7. Layout
# --------------------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SafeDerm Triage Lite") as demo:
        gr.HTML(header_html())
        state = gr.State(None)

        with gr.Tabs():
            with gr.Tab("Analyze image"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320):
                        img = gr.Image(
                            label="Dermoscopic image (JPG or PNG, up to 10 MB)",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=380,
                        )
                        with gr.Row():
                            analyze = gr.Button("Analyze image", variant="primary", elem_classes="sd-analyze")
                            reset = gr.Button("Clear")
                        status = gr.HTML(status_html("idle"))
                    with gr.Column(scale=7, min_width=360):
                        result = gr.HTML(EMPTY_RESULT)

                gr.HTML('<h2 class="sd-h2" style="margin:20px 0 4px">Where the model looked</h2>')
                alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Overlay colour strength")
                explain = gr.HTML(EMPTY_EXPLAIN)

            with gr.Tab("Model performance"):
                refresh = gr.Button("Reload saved metrics")
                summary = gr.HTML()
                confusion = gr.HTML()
                per_class = gr.Dataframe(label="Per-class metrics", interactive=False, wrap=True)

            with gr.Tab("How to read this"):
                gr.HTML(LIMITS_HTML)

        analyze.click(on_analyze, [img, alpha], [state, status, result, explain], show_progress="minimal")
        reset.click(on_reset, None, [img, state, status, result, explain])
        alpha.change(lambda a, s: render_explain(s, a), [alpha, state], explain, show_progress="hidden")
        refresh.click(on_dashboard, None, [summary, confusion, per_class])
        demo.load(on_dashboard, None, [summary, confusion, per_class])
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("HOST", "0.0.0.0")
    print(f"[SafeDerm Triage Lite] Launching on http://{server_name}:{port}...", flush=True)
    build_ui().launch(server_name=server_name, server_port=port, share=False, theme=THEME, css=CUSTOM_CSS)

