# Generator for C:\SafeDerm\src\app.py
# Redesigned Clean, Professional, Zero-Overlap Interface for SafeDerm Triage Lite

import os

app_path = r"C:\SafeDerm\src\app.py"

app_code = '''# SafeDerm Triage Lite - Main Application
# Educational/research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.

import os
import sys
import html
import gradio as gr
from PIL import Image
import pandas as pd

# Add src and saved_model to sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import InferenceService, CLASS_NAMES, FULL_CLASS_NAMES
from gradcam import GradCAM
import dashboard

# Initialize single InferenceService instance
service = InferenceService.get_instance()
gradcam_service = GradCAM(service.model)

# Medical disclaimer text
DISCLAIMER_TEXT = "Research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician."

# Modern Clinical Theme
custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
).set(
    body_background_fill="#f8fafc",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_radius="10px",
    block_shadow="0 1px 3px 0 rgba(0, 0, 0, 0.03)",
)

CUSTOM_CSS = """
/* SafeDerm Triage Lite - Streamlined Clinical Styling */
:root {
    --primary: #0f172a;
    --accent: #0284c7;
    --border: #e2e8f0;
    --bg-main: #f8fafc;
}

body, .gradio-container {
    background-color: var(--bg-main) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #1e293b !important;
}

/* Header & Disclaimers */
.disclaimer-banner {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: #92400e;
    font-weight: 500;
}

.header-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 22px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-brand h1 {
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.2px;
}

.header-brand p {
    font-size: 13px;
    color: #64748b;
    margin: 3px 0 0 0;
}

.badge-pill {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 16px;
    padding: 5px 12px;
    font-size: 12px;
    color: #334155;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
}

.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10b981;
}

/* Segmented Tab Navigation */
.tabs > .tab-nav, div[role="tablist"] {
    background: #f1f5f9 !important;
    padding: 5px !important;
    border-radius: 10px !important;
    gap: 6px !important;
    border: 1px solid #e2e8f0 !important;
    margin-bottom: 16px !important;
}

.tabs > .tab-nav > button, button[role="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 8px 20px !important;
    border: none !important;
    color: #475569 !important;
    background: transparent !important;
    transition: all 0.15s ease !important;
}

.tabs > .tab-nav > button.selected, button[role="tab"][aria-selected="true"], button[role="tab"].selected {
    background: #0f172a !important;
    color: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
}

.tabs > .tab-nav > button:hover:not(.selected) {
    background: #e2e8f0 !important;
    color: #0f172a !important;
}

/* Status Pills */
.status-pill-idle {
    font-size: 12px;
    color: #64748b;
    font-weight: 500;
    text-align: center;
    padding: 7px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-top: 8px;
}

.status-pill-success {
    font-size: 12px;
    color: #047857;
    font-weight: 600;
    text-align: center;
    padding: 7px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 6px;
    margin-top: 8px;
}

.status-pill-error {
    font-size: 12px;
    color: #b91c1c;
    font-weight: 600;
    text-align: center;
    padding: 7px;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    margin-top: 8px;
}

/* CAM Visual Titles */
.cam-title {
    font-size: 11.5px;
    font-weight: 700;
    color: #475569;
    text-align: center;
    padding: 5px 8px;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* KPI Cards */
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

.stat-card .val {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 4px;
}

.stat-card .lbl {
    font-size: 11.5px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Top 3 Progress Bars */
.bar-container {
    margin: 8px 0;
}

.bar-header {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    font-weight: 600;
    margin-bottom: 4px;
    color: #1e293b;
}

.bar-track {
    background: #e2e8f0;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease;
}
"""


def analyze_image(img_input):
    """
    Main analysis pipeline triggered by Analyze Image button.
    Returns:
    - result_summary_html
    - top3_bars_html
    - full_table_df
    - orig_cropped_img
    - heatmap_img
    - overlay_img
    - status_html
    """
    if img_input is None:
        err_html = """
        <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:20px; color:#991b1b; font-size:13.5px; text-align:center;">
          <strong>Input Required:</strong> Please upload a dermoscopic skin lesion image on the left panel before clicking Analyze.
        </div>
        """
        status_err = "<div class='status-pill-error'>&times; No image uploaded</div>"
        return err_html, "", None, None, None, None, status_err

    res = service.predict(img_input)
    if not res["success"]:
        err_msg = html.escape(res["error"])
        err_html = f"""
        <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:18px; color:#991b1b; font-size:13px;">
          <strong>Validation Notice:</strong> {err_msg}
        </div>
        """
        status_err = f"<div class='status-pill-error'>&times; Validation failed: {err_msg}</div>"
        return err_html, "", None, None, None, None, status_err

    top_class = res["top_class"]
    top_full = res["top_full_name"]
    top_score = res["top_score"]
    
    # Categorization styling
    if top_class == "MEL":
        cat_badge = "MALIGNANT ALERT &bull; HIGH CLINICAL SUSPICION"
        cat_border = "#fca5a5"
        cat_bg = "#fef2f2"
        cat_text = "#991b1b"
        tag_bg = "#dc2626"
    elif top_class in ["BCC", "SCC", "AK"]:
        cat_badge = "CARCINOMA / PRE-MALIGNANT CATEGORY"
        cat_border = "#fde68a"
        cat_bg = "#fffbeb"
        cat_text = "#92400e"
        tag_bg = "#d97706"
    else:
        cat_badge = "BENIGN LESION MATCH"
        cat_border = "#bbf7d0"
        cat_bg = "#f0fdf4"
        cat_text = "#166534"
        tag_bg = "#16a34a"

    summary_html = f"""
    <div style="background:{cat_bg}; border:1px solid {cat_border}; border-radius:10px; padding:16px 20px; margin-bottom:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:11px; font-weight:800; color:{cat_text}; text-transform:uppercase; letter-spacing:0.7px;">
          Highest-Scoring Model Class &bull; {cat_badge}
        </span>
        <span style="background:{tag_bg}; color:#ffffff; font-size:12px; font-weight:700; padding:3px 10px; border-radius:14px;">
          {top_score*100:.1f}% Score
        </span>
      </div>
      <div style="font-size:22px; font-weight:800; color:#0f172a; margin-top:6px;">
        {top_class} &mdash; {top_full}
      </div>
      <div style="font-size:12px; color:#64748b; margin-top:5px;">
        EfficientNet-B0 + CBAM (Epoch {res['epoch']}) &bull; Softmax Score Output &bull; Hardware: {res['device']}
      </div>
    </div>
    """

    # Top-3 Bars
    bars_html = '<div style="margin-bottom:14px; background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px 18px;">'
    bars_html += '<div style="font-size:12px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Top-3 Ranked Classes</div>'
    
    colors = ["#0284c7", "#0d9488", "#d97706"]
    for idx, item in enumerate(res["top3"]):
        c_code = item["class"]
        c_name = item["full_name"]
        pct = item["score"] * 100
        bar_color = colors[idx % len(colors)]
        rank_num = idx + 1
        bars_html += f"""
        <div class="bar-container">
          <div class="bar-header">
            <span>
              <strong style="color:#64748b; margin-right:4px;">#{rank_num}</strong> 
              {c_code} &ndash; {c_name}
            </span>
            <span style="font-weight:700; color:#0f172a;">{pct:.1f}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div>
          </div>
        </div>
        """
    bars_html += '</div>'

    # Full 8-Class table
    table_rows = []
    for c in CLASS_NAMES:
        table_rows.append({
            "Class Code": c,
            "Category Name": FULL_CLASS_NAMES[c],
            "Softmax Score": f"{res['scores'][c]:.4f}",
            "Score (%)": f"{res['scores'][c]*100:.2f}%"
        })
    df_scores = pd.DataFrame(table_rows)

    # Generate Grad-CAM for top class
    top_class_idx = CLASS_NAMES.index(top_class)
    heatmap_img, overlay_img, cam_err = gradcam_service.generate(
        input_tensor=res["input_tensor"],
        target_class_idx=top_class_idx,
        cropped_pil=res["cropped_img"]
    )

    cropped_img = res["cropped_img"]
    status_html = f"<div class='status-pill-success'>&check; Inference complete &bull; Top Class: {top_class} ({top_score*100:.1f}%) &bull; Grad-CAM Target: Index {top_class_idx} ({top_class}) Logit</div>"
    if cam_err is not None:
        status_html += f"<div style='font-size:11px; color:#d97706; text-align:center;'>Grad-CAM notice: {cam_err}</div>"

    return summary_html, bars_html, df_scores, cropped_img, heatmap_img, overlay_img, status_html


def clear_analysis():
    """Resets all outputs in UI A."""
    empty_html = """
    <div style="background:#ffffff; border:1px dashed #cbd5e1; border-radius:10px; padding:36px 20px; text-align:center; color:#64748b;">
      <div style="font-size:15px; font-weight:700; color:#334155; margin-bottom:4px;">No Image Analyzed</div>
      <div style="font-size:13px; max-width:380px; margin:0 auto;">
        Upload a dermoscopic lesion image on the left and click <strong>Analyze Image</strong> to view predictions and visual explainability.
      </div>
    </div>
    """
    idle_status = "<div class='status-pill-idle'>Ready for image upload</div>"
    return empty_html, "", None, None, None, None, idle_status


# Build Gradio Blocks Application
with gr.Blocks(title="SafeDerm Triage Lite") as demo:
    # 1. Top Mandatory Clinical Notice
    gr.HTML(f"""
    <div class="disclaimer-banner">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#b45309" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
      </svg>
      <span><strong>Clinical Research Notice:</strong> {DISCLAIMER_TEXT}</span>
    </div>
    """)

    # 2. Main Clinical Header Card
    gr.HTML("""
    <div class="header-card">
      <div class="header-brand">
        <h1>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
          </svg>
          SafeDerm Triage Lite
        </h1>
        <p>Explainable AI for Educational Skin Lesion Classification &bull; ISIC 2019 Research Showcase</p>
      </div>
      <div style="display:flex; gap:8px;">
        <div class="badge-pill">
          <div class="pulse-dot"></div>
          <span>Model Ready (RTX 4050)</span>
        </div>
        <div class="badge-pill">
          <span>Epoch 22 Checkpoint</span>
        </div>
      </div>
    </div>
    """)

    # 3. Two Distinct Tabs
    with gr.Tabs() as tabs:
        # =============================================================
        # TAB 1: Image Analysis Studio
        # =============================================================
        with gr.Tab("Image Analysis Studio", id="studio_tab"):
            with gr.Row():
                # Left Column: Upload & Controls
                with gr.Column(scale=5):
                    with gr.Group():
                        gr.HTML("""
                        <div style="margin-bottom:8px;">
                          <div style="font-size:15px; font-weight:700; color:#0f172a;">Upload Dermoscopic Image</div>
                          <div style="font-size:12px; color:#64748b;">Supports JPG, PNG (≤10 MB) &bull; Auto-cropped to 224&times;224</div>
                        </div>
                        """)
                        image_input = gr.Image(
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=290,
                            show_label=False,
                            buttons=[]
                        )
                        with gr.Row():
                            analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
                            clear_btn = gr.Button("Reset", variant="secondary", size="lg")

                        status_bar = gr.HTML("<div class='status-pill-idle'>Ready for image upload</div>")

                # Right Column: Predictions & Visual Explainability
                with gr.Column(scale=7):
                    # Results Summary Card
                    results_html = gr.HTML("""
                    <div style="background:#ffffff; border:1px dashed #cbd5e1; border-radius:10px; padding:36px 20px; text-align:center; color:#64748b;">
                      <div style="font-size:15px; font-weight:700; color:#334155; margin-bottom:4px;">No Image Analyzed</div>
                      <div style="font-size:13px; max-width:380px; margin:0 auto;">
                        Upload a dermoscopic lesion image on the left and click <strong>Analyze Image</strong> to view predictions and visual explainability.
                      </div>
                    </div>
                    """)

                    top3_bars = gr.HTML("")

                    # Grad-CAM Attention Visuals (Clean titles, no overlapping buttons)
                    with gr.Group():
                        gr.HTML("<div style='font-size:13px; font-weight:700; color:#0f172a; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.5px;'>Model Explainability (Grad-CAM Feature Map)</div>")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                gr.HTML("<div class='cam-title'>1. Preprocessed Input</div>")
                                cam_orig = gr.Image(interactive=False, height=200, show_label=False, buttons=[])
                            with gr.Column(scale=1, min_width=100):
                                gr.HTML("<div class='cam-title'>2. Sensitivity Map</div>")
                                cam_heat = gr.Image(interactive=False, height=200, show_label=False, buttons=[])
                            with gr.Column(scale=1, min_width=100):
                                gr.HTML("<div class='cam-title'>3. Overlay Blend</div>")
                                cam_over = gr.Image(interactive=False, height=200, show_label=False, buttons=[])
                        
                        gr.HTML("""
                        <div style="display:flex; align-items:center; justify-content:center; gap:10px; margin:10px 0 6px 0;">
                          <span style="font-size:11px; font-weight:600; color:#64748b;">Low Relative Activation (0.0)</span>
                          <div style="width:160px; height:10px; border-radius:5px; background:linear-gradient(to right, #000080, #00ffff, #ffff00, #ff0000); border:1px solid #cbd5e1;"></div>
                          <span style="font-size:11px; font-weight:600; color:#64748b;">High Relative Activation (1.0)</span>
                        </div>
                        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; font-size:12px; color:#475569; margin-top:6px; line-height:1.45;">
                          <strong>Explanatory Caption:</strong> Grad-CAM visualizes image regions associated with the selected class score. Highlighted regions are not proof of medically meaningful reasoning, and the map may be influenced by image artifacts.
                          <br>
                          <span style="color:#64748b; font-size:11.5px; font-style:italic;">Grad-CAM is an approximate visualization of model sensitivity. It does not establish clinical validity and may highlight imaging artifacts.</span>
                        </div>
                        """)

                    # Full 8-Class Scores Breakdown
                    with gr.Accordion("Complete 8-Class Raw Softmax Breakdown", open=False):
                        full_table = gr.Dataframe(
                            headers=["Class Code", "Category Name", "Softmax Score", "Score (%)"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False
                        )

            # Event handlers
            analyze_btn.click(
                fn=analyze_image,
                inputs=[image_input],
                outputs=[results_html, top3_bars, full_table, cam_orig, cam_heat, cam_over, status_bar]
            )
            clear_btn.click(
                fn=clear_analysis,
                inputs=[],
                outputs=[results_html, top3_bars, full_table, cam_orig, cam_heat, cam_over, status_bar]
            )

        # =============================================================
        # TAB 2: Model Insights & Performance Dashboard
        # =============================================================
        with gr.Tab("Model Insights & Performance Dashboard", id="dashboard_tab"):
            gr.HTML("""
            <div style="margin-bottom:14px;">
              <div style="font-size:16px; font-weight:700; color:#0f172a;">Evaluated Model Performance (ISIC 2019 Test Baseline)</div>
              <div style="font-size:12.5px; color:#64748b;">Frozen baseline evaluation conducted on held-out ISIC 2019 test set (2,534 images). Loaded directly from results artifacts.</div>
            </div>
            """)

            # 4 KPI Stat Cards
            metrics_data = dashboard.load_test_metrics()
            acc_str = f"{metrics_data.get('accuracy', 0.8216)*100:.1f}%"
            f1_str = f"{metrics_data.get('macro_f1', 0.7606)*100:.1f}%"
            bal_str = f"{metrics_data.get('balanced_accuracy', 0.7447)*100:.1f}%"
            mel_rec_str = f"{metrics_data.get('per_class', {}).get('MEL', {}).get('recall', 0.6416)*100:.1f}%"

            with gr.Row():
                with gr.Column(scale=1, min_width=120):
                    gr.HTML(f"""
                    <div class="stat-card">
                      <div class="val">{acc_str}</div>
                      <div class="lbl">Overall Test Accuracy</div>
                    </div>
                    """)
                with gr.Column(scale=1, min_width=120):
                    gr.HTML(f"""
                    <div class="stat-card">
                      <div class="val">{bal_str}</div>
                      <div class="lbl">Balanced Accuracy</div>
                    </div>
                    """)
                with gr.Column(scale=1, min_width=120):
                    gr.HTML(f"""
                    <div class="stat-card">
                      <div class="val">{f1_str}</div>
                      <div class="lbl">Macro-F1 Score</div>
                    </div>
                    """)
                with gr.Column(scale=1, min_width=120):
                    gr.HTML(f"""
                    <div class="stat-card" style="border-color:#fecaca; background:#fff5f5;">
                      <div class="val" style="color:#b91c1c;">{mel_rec_str}</div>
                      <div class="lbl" style="color:#991b1b;">Melanoma Recall</div>
                    </div>
                    """)

            # Confusion Matrix & Class Performance
            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML("<div style='font-size:13px; font-weight:700; color:#0f172a; margin:14px 0 8px 0;'>Held-Out Test Set Confusion Matrix (8 Classes)</div>")
                    cm_img_val = dashboard.get_confusion_matrix_image()
                    gr.Image(value=cm_img_val, interactive=False, show_label=False, buttons=[])
                with gr.Column(scale=6):
                    gr.HTML("<div style='font-size:13px; font-weight:700; color:#0f172a; margin:14px 0 8px 0;'>Per-Class Recall & F1-Score Breakdown</div>")
                    perf_img_val = dashboard.get_class_performance_plot()
                    gr.Image(value=perf_img_val, interactive=False, show_label=False, buttons=[])

            # Architecture Explorer
            gr.HTML("<div style='font-size:14px; font-weight:700; color:#0f172a; margin:20px 0 10px 0;'>Model Architecture: EfficientNet-B0 + CBAM Attention</div>")
            gr.HTML(dashboard.get_architecture_svg())

            # Dataset & Limitations Panel
            gr.HTML("<div style='font-size:14px; font-weight:700; color:#0f172a; margin:20px 0 10px 0;'>Dataset Characteristics & Research Limitations</div>")
            gr.HTML("""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:18px 22px; font-size:13px; line-height:1.6; color:#334155;">
              <ul style="margin:0; padding-left:18px;">
                <li style="margin-bottom:6px;"><strong>Class Imbalance:</strong> Melanocytic Nevi (NV) comprises >50% of images, while Dermatofibroma (DF) and Vascular Lesions (VASC) represent <1% each.</li>
                <li style="margin-bottom:6px;"><strong>Melanoma Sensitivity Gap:</strong> Test melanoma recall is 64.2%. False negatives remain a primary safety challenge for educational discussion.</li>
                <li style="margin-bottom:6px;"><strong>Internal Test Evaluation:</strong> Metrics represent internal ISIC 2019 held-out splits. Generalization to hospital hardware or diverse skin tones (Fitzpatrick IV-VI) requires external validation.</li>
                <li><strong>Shortcut Learning:</strong> Grad-CAM highlights potential reliance on dermatoscope borders or vignetting rather than pathological morphology.</li>
              </ul>
            </div>
            """)

    # 4. Bottom Mandatory Clinical Notice
    gr.HTML(f"""
    <div class="disclaimer-banner" style="margin-top:20px; margin-bottom:0;">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#b45309" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
      </svg>
      <span><strong>Educational Research Disclaimer:</strong> {DISCLAIMER_TEXT}</span>
    </div>
    """)

if __name__ == "__main__":
    print("[SafeDerm Triage Lite] Launching on http://127.0.0.1:7860...", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS, theme=custom_theme)
'''

with open(app_path, "w", encoding="utf-8") as f:
    f.write(app_code)
print(f"Created {app_path}")
