# SafeDerm Triage Lite - Model Insights & Performance Analytics Dashboard
# Educational/research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import Dict, Any, Optional

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
FULL_CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma",
}


def load_test_metrics() -> Dict[str, Any]:
    """Loads baseline evaluated test metrics from test_metrics.json."""
    metrics_path = os.path.join(RESULTS_DIR, "test_metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading test_metrics.json: {e}")
        return {}


def get_confusion_matrix_image() -> Optional[Image.Image]:
    """Returns PIL Image of the recorded confusion matrix artifact."""
    png_path = os.path.join(RESULTS_DIR, "confusion_matrix_test.png")
    if os.path.exists(png_path):
        try:
            return Image.open(png_path)
        except Exception:
            pass

    # Fallback: Render from CSV
    csv_path = os.path.join(RESULTS_DIR, "confusion_matrix_test.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col=0)
            fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
            im = ax.imshow(df.values, cmap="Blues", interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(CLASS_NAMES, fontsize=9)
            ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold", labelpad=8)
            ax.set_ylabel("True Label", fontsize=10, fontweight="bold", labelpad=8)
            ax.set_title("Test Set Confusion Matrix (2,534 images)", fontsize=11, fontweight="bold", pad=12)

            # Annotate numbers
            thresh = df.values.max() / 2.0
            for i in range(len(CLASS_NAMES)):
                for j in range(len(CLASS_NAMES)):
                    val = df.values[i, j]
                    ax.text(j, i, f"{int(val)}", ha="center", va="center",
                            color="white" if val > thresh else "black", fontsize=8)

            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            print(f"Error rendering confusion matrix from CSV: {e}")
            return None

    return None


def get_class_performance_plot() -> Optional[Image.Image]:
    """Generates horizontal bar chart of per-class Recall and F1-score."""
    metrics = load_test_metrics()
    if not metrics or "per_class" not in metrics:
        return None

    per_class = metrics["per_class"]
    classes = CLASS_NAMES
    recalls = [per_class[c]["recall"] * 100 for c in classes]
    f1s = [per_class[c]["f1-score"] * 100 for c in classes]
    supports = [int(per_class[c]["support"]) for c in classes]

    y = np.arange(len(classes))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=120)
    rects1 = ax.barh(y - height/2, recalls, height, label="Recall (%)", color="#1d4ed8")
    rects2 = ax.barh(y + height/2, f1s, height, label="F1-Score (%)", color="#0d9488")

    ax.set_xlabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_title("Per-Class Test Performance (Recorded Baseline)", fontsize=11, fontweight="bold", pad=10)
    ax.set_yticks(y)
    # Add support count to class label
    labels_with_support = [f"{c} (n={supports[i]})" for i, c in enumerate(classes)]
    ax.set_yticklabels(labels_with_support, fontsize=9)
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate bars
    for r in rects1:
        w = r.get_width()
        ax.text(w + 1, r.get_y() + r.get_height()/2, f"{w:.1f}%", va="center", fontsize=7.5, color="#1e3a8a")
    for r in rects2:
        w = r.get_width()
        ax.text(w + 1, r.get_y() + r.get_height()/2, f"{w:.1f}%", va="center", fontsize=7.5, color="#115e59")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def get_architecture_svg() -> str:
    """Returns SVG diagram representing the EfficientNet-B0 + CBAM architecture."""
    return """
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; font-family:system-ui, -apple-system, sans-serif;">
      <h4 style="margin:0 0 12px 0; color:#1e293b; font-size:15px; font-weight:700;">Model Architecture: EfficientNet-B0 with CBAM Attention</h4>
      <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px; font-size:12px;">
        <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px; padding:10px 14px; text-align:center; min-width:110px;">
          <div style="font-weight:700; color:#0f172a;">Input Image</div>
          <div style="color:#64748b; font-size:11px;">256x256 Bicubic</div>
          <div style="color:#64748b; font-size:11px;">CenterCrop 224x224</div>
        </div>
        <div style="color:#94a3b8; font-weight:700;">&rarr;</div>
        <div style="background:#e0f2fe; border:1px solid #7dd3fc; border-radius:8px; padding:10px 14px; text-align:center; min-width:110px;">
          <div style="font-weight:700; color:#0369a1;">Stages 0 &ndash; 2</div>
          <div style="color:#0284c7; font-size:11px;">Stem + MBConv</div>
          <div style="color:#64748b; font-size:11px;">24 ch, 56x56</div>
        </div>
        <div style="color:#94a3b8; font-weight:700;">&rarr;</div>
        <div style="background:#dbeafe; border:1px solid #93c5fd; border-radius:8px; padding:10px 14px; text-align:center; min-width:125px;">
          <div style="font-weight:700; color:#1d4ed8;">Stage 3 + CBAM</div>
          <div style="color:#2563eb; font-size:11px;">40 ch, 28x28</div>
          <div style="color:#475569; font-size:10px;">Channel + Spatial Attn</div>
        </div>
        <div style="color:#94a3b8; font-weight:700;">&rarr;</div>
        <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px; padding:10px 14px; text-align:center; min-width:90px;">
          <div style="font-weight:700; color:#0f172a;">Stage 4</div>
          <div style="color:#64748b; font-size:11px;">80 ch, 14x14</div>
        </div>
        <div style="color:#94a3b8; font-weight:700;">&rarr;</div>
        <div style="background:#ccfbf1; border:2px solid #0d9488; border-radius:8px; padding:10px 14px; text-align:center; min-width:135px; box-shadow:0 2px 4px rgba(13,148,136,0.1);">
          <div style="font-weight:800; color:#0f766e;">Stage 5 + CBAM</div>
          <div style="color:#0d9488; font-size:11px; font-weight:600;">112 ch, 14x14</div>
          <div style="color:#047857; font-size:10px; font-weight:700;">&larr; Grad-CAM Target</div>
        </div>
        <div style="color:#94a3b8; font-weight:700;">&rarr;</div>
        <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px; padding:10px 14px; text-align:center; min-width:90px;">
          <div style="font-weight:700; color:#0f172a;">Stage 6</div>
          <div style="color:#64748b; font-size:11px;">192 ch, 7x7</div>
        </div>
        <div style="color:#94a3b8; font-weight:700;">&rarr;</div>
        <div style="background:#dbeafe; border:1px solid #93c5fd; border-radius:8px; padding:10px 14px; text-align:center; min-width:125px;">
          <div style="font-weight:700; color:#1d4ed8;">Stage 7 + CBAM</div>
          <div style="color:#2563eb; font-size:11px;">320 ch, 7x7</div>
          <div style="color:#475569; font-size:10px;">Attention Stage</div>
        </div>
        <div style="color:#94a3b8; font-weight:700;">&rarr;</div>
        <div style="background:#fef3c7; border:1px solid #fde68a; border-radius:8px; padding:10px 14px; text-align:center; min-width:110px;">
          <div style="font-weight:700; color:#b45309;">Classifier (8)</div>
          <div style="color:#92400e; font-size:11px;">Linear Head</div>
          <div style="color:#64748b; font-size:10px;">ISIC Classes 0..7</div>
        </div>
      </div>
      <div style="margin-top:14px; font-size:11.5px; color:#64748b; line-height:1.5;">
        <strong>Verified Parameter Count:</strong> 4,033,797 trainable parameters | 4,075,862 state-dictionary elements (includes 42,065 BatchNorm tracking buffers) | Checkpoint Epoch 22
      </div>
    </div>
    """
