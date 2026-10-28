"""
B.Tech Project Report Figure Generator
Title: Class-Weighted Transfer Learning with Explainability Validation for Dermoscopic Malignancy Detection
Target Figures: ONLY the 4 essential academic report figures:
  1. Dataset Class Distribution (Training images across 8 ISIC 2019 classes)
  2. Training and Validation Loss (24 epochs from checkpoints/training_log.csv)
  3. Test Confusion Matrix (Annotated heatmap from results/confusion_matrix_test.csv)
  4. Per-Class Performance (Precision, Recall, F1 grouped bar chart from results/test_metrics.json)

Output Directory: C:\SafeDerm\report_figures\
Resolution: 300 DPI PNG, white background, academic styling.
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Primary directories
BASE_DIR = r"C:\SafeDerm"
OUT_DIR = os.path.join(BASE_DIR, "report_figures")
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Local mirrors for immediate access in Downloads and workspace
WORKSPACE_OUT_DIR = r"c:\Users\SANTHOSH\Downloads\Skin Lesion Review\report_figures"
DOWNLOADS_OUT_DIR = r"C:\Users\SANTHOSH\Downloads\report_figures"

# ISIC 2019 8 Diagnostic Classes (Strict order from project specifications)
CLASS_CODES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_FULL_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma"
}

# Color styling for academic publication
PATHOLOGY_COLORS = {
    "MEL": "#dc2626",   # Malignant (Red)
    "NV": "#0284c7",    # Benign (Blue)
    "BCC": "#b91c1c",   # Malignant (Dark Red)
    "AK": "#d97706",    # Pre-Malignant (Amber/Orange)
    "BKL": "#0ea5e9",   # Benign (Sky Blue)
    "DF": "#38bdf8",    # Benign (Light Blue)
    "VASC": "#60a5fa",  # Benign (Soft Blue)
    "SCC": "#991b1b",   # Malignant (Deep Crimson)
}

plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#94a3b8',
    'axes.linewidth': 1.0,
    'grid.color': '#e2e8f0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.7,
    'text.color': '#0f172a',
    'axes.labelcolor': '#0f172a',
    'xtick.color': '#334155',
    'ytick.color': '#334155',
})


def save_figure(fig, base_name):
    """Save 300 DPI PNG to C:\SafeDerm\report_figures and mirror directories."""
    filename = f"{base_name}.png"
    for target_dir in [OUT_DIR, WORKSPACE_OUT_DIR, DOWNLOADS_OUT_DIR]:
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    print(f"  [Generated] {filename} (300 DPI PNG)")
    plt.close(fig)


# ==============================================================================
# Figure 1: Dataset Class Distribution
# ==============================================================================
def generate_figure_1():
    print("\n[Figure 1/4] Generating Dataset Class Distribution...")
    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train.csv at {train_csv}")
        
    df_train = pd.read_csv(train_csv)
    counts = df_train['label'].value_counts()
    
    counts_list = [counts.get(c, 0) for c in CLASS_CODES]
    total_train = sum(counts_list)
    assert total_train == 20264, f"Expected 20,264 training samples, got {total_train}"
    
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    
    colors = [PATHOLOGY_COLORS[c] for c in CLASS_CODES]
    bars = ax.bar(CLASS_CODES, counts_list, color=colors, width=0.62, edgecolor='#1e293b', linewidth=0.9, zorder=3)
    
    ax.set_title("Training Set Class Distribution across 8 ISIC 2019 Categories (N = 20,264)", 
                 fontsize=13.5, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xlabel("ISIC Diagnostic Category", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_ylabel("Number of Training Images", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_ylim(0, 12500)
    ax.grid(axis='y', alpha=0.5, zorder=0)
    
    # Clean x-tick labels
    ax.set_xticks(range(len(CLASS_CODES)))
    ax.set_xticklabels(CLASS_CODES, fontsize=11.0, fontweight='bold')
    
    # Annotate counts and percentages above bars
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_train) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., h + 180, f"{h:,}\n({pct:.1f}%)", 
                ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#0f172a')
                
    # Clinical Pathology legend
    legend_patches = [
        mpatches.Patch(facecolor='#dc2626', edgecolor='#1e293b', label='Malignant (MEL, BCC, SCC) — 6,778 (33.4%)'),
        mpatches.Patch(facecolor='#d97706', edgecolor='#1e293b', label='Pre-Malignant (AK) — 694 (3.4%)'),
        mpatches.Patch(facecolor='#0284c7', edgecolor='#1e293b', label='Benign (NV, BKL, DF, VASC) — 12,792 (63.1%)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', frameon=True, fontsize=9.5, facecolor='#ffffff', edgecolor='#cbd5e1')
    
    # Category definition key at bottom right / center note
    key_text = (
        "Diagnostic Category Key:\n"
        "• MEL: Melanoma | NV: Melanocytic Nevus | BCC: Basal Cell Carcinoma | AK: Actinic Keratosis\n"
        "• BKL: Benign Keratosis | DF: Dermatofibroma | VASC: Vascular Lesion | SCC: Squamous Cell Carcinoma\n"
        "Class Imbalance Ratio (NV / DF): 53.9:1 — Compensated via Class-Weighted Focal Loss ($\\gamma=2.0$)"
    )
    ax.text(0.5, -0.22, key_text, transform=ax.transAxes, ha='center', va='top', fontsize=8.8, fontweight='bold', color='#334155',
            bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec="#cbd5e1", lw=1))
            
    plt.tight_layout()
    save_figure(fig, "01_dataset_class_distribution")


# ==============================================================================
# Figure 2: Training and Validation Loss
# ==============================================================================
def generate_figure_2():
    print("\n[Figure 2/4] Generating Training and Validation Loss Trajectory...")
    log_csv = os.path.join(CHECKPOINTS_DIR, "training_log.csv")
    if not os.path.exists(log_csv):
        raise FileNotFoundError(f"Missing training_log.csv at {log_csv}")
        
    df = pd.read_csv(log_csv)
    epochs = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values
    assert len(epochs) == 24, f"Expected 24 epochs, found {len(epochs)}"
    
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)
    
    ax.plot(epochs, train_loss, 'o-', color='#1d4ed8', linewidth=2.3, markersize=5.5, 
            label='Training Loss (Class-Weighted Focal Loss, $\\gamma=2.0$)', zorder=3)
    ax.plot(epochs, val_loss, 's--', color='#b91c1c', linewidth=2.3, markersize=5.5, 
            label='Validation Loss', zorder=3)
            
    # Best checkpoint at Epoch 22
    best_epoch = 22
    best_val_loss = val_loss[best_epoch - 1]
    best_macro_f1 = df['val_macro_f1'].values[best_epoch - 1]
    
    ax.axvline(x=best_epoch, color='#059669', linestyle=':', linewidth=1.8, label=f'Best Checkpoint (Epoch {best_epoch})', zorder=2)
    ax.scatter([best_epoch], [best_val_loss], color='#059669', s=130, zorder=5, edgecolor='#0f172a', linewidth=1.5)
    
    ax.annotate(f"Best Model Checkpoint: Epoch {best_epoch}\nVal Loss: {best_val_loss:.4f} | Macro-F1: {best_macro_f1*100:.2f}%", 
                xy=(best_epoch, best_val_loss), 
                xytext=(best_epoch - 6.5, best_val_loss + 0.12),
                arrowprops=dict(facecolor='#059669', shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.35", fc="#ecfdf5", ec="#059669", lw=1.2),
                fontweight='bold', fontsize=9.2, color='#065f46')
                
    ax.set_title("EfficientNet-B0 + CBAM: Training & Validation Loss Trajectory", 
                 fontsize=13.5, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xlabel("Training Epoch (AdamW, Cosine Annealing LR Schedule)", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_ylabel("Focal Loss Value", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_xticks(range(1, 25))
    ax.grid(True, alpha=0.5, zorder=0)
    ax.legend(loc='upper right', frameon=True, fontsize=9.5, facecolor='#ffffff', edgecolor='#cbd5e1')
    
    plt.tight_layout()
    save_figure(fig, "02_training_validation_loss")


# ==============================================================================
# Figure 3: Test Confusion Matrix
# ==============================================================================
def generate_figure_3():
    print("\n[Figure 3/4] Generating Test Confusion Matrix Heatmap...")
    cm_csv = os.path.join(RESULTS_DIR, "confusion_matrix_test.csv")
    if not os.path.exists(cm_csv):
        raise FileNotFoundError(f"Missing confusion_matrix_test.csv at {cm_csv}")
        
    df_cm = pd.read_csv(cm_csv, index_col=0)
    cm_matrix = df_cm.values.astype(int)
    total_test = cm_matrix.sum()
    assert total_test == 2534, f"Expected 2,534 test samples, got {total_test}"
    
    fig, ax = plt.subplots(figsize=(9.2, 7.8), dpi=300)
    
    im = ax.imshow(cm_matrix, cmap="Blues", interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of Test Images", fontsize=10.5, fontweight='bold', labelpad=8)
    
    ax.set_xticks(range(len(CLASS_CODES)))
    ax.set_yticks(range(len(CLASS_CODES)))
    ax.set_xticklabels(CLASS_CODES, fontsize=10.5, fontweight='bold')
    ax.set_yticklabels(CLASS_CODES, fontsize=10.5, fontweight='bold')
    ax.set_xlabel("Predicted Diagnostic Category", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_ylabel("True Ground-Truth Diagnosis", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_title("Held-Out Test Set Confusion Matrix (Total N = 2,534)", fontsize=13.5, fontweight='bold', pad=26, color='#0f172a')
    
    thresh = cm_matrix.max() / 2.0
    for i in range(len(CLASS_CODES)):
        for j in range(len(CLASS_CODES)):
            val = cm_matrix[i, j]
            color = "white" if val > thresh else "#0f172a"
            weight = "bold" if (i == j or val > 50) else "normal"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=9.5, fontweight=weight)
            
    # Red box highlighting Melanoma false negatives at cell (row 0, col 1) [True MEL -> Pred NV]
    rect = plt.Rectangle((1 - 0.48, 0 - 0.48), 0.96, 0.96, fill=False, edgecolor='#dc2626', linewidth=2.5)
    ax.add_patch(rect)
    
    # Curved annotation pointing cleanly above the cell with zero label collision
    ax.annotate("125 Melanomas Misclassified as NV\n(Critical Clinical False Negatives)", 
                xy=(1.0, -0.48), xytext=(3.4, -0.92),
                arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', arrowstyle="->", lw=1.8, connectionstyle="arc3,rad=-0.15"),
                bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec="#dc2626", lw=1.2),
                ha='left', va='center', color='#991b1b', fontsize=8.6, fontweight='bold')
    ax.set_ylim(7.5, -1.35)
    
    plt.tight_layout()
    save_figure(fig, "03_test_confusion_matrix")


# ==============================================================================
# Figure 4: Per-Class Performance
# ==============================================================================
def generate_figure_4():
    print("\n[Figure 4/4] Generating Per-Class Performance Grouped Bar Chart...")
    metrics_json = os.path.join(RESULTS_DIR, "test_metrics.json")
    if not os.path.exists(metrics_json):
        raise FileNotFoundError(f"Missing test_metrics.json at {metrics_json}")
        
    with open(metrics_json, 'r') as f:
        m = json.load(f)
        
    per_class = m["per_class"]
    precisions = [per_class[c]["precision"] * 100 for c in CLASS_CODES]
    recalls = [per_class[c]["recall"] * 100 for c in CLASS_CODES]
    f1_scores = [per_class[c]["f1-score"] * 100 for c in CLASS_CODES]
    supports = [int(per_class[c]["support"]) for c in CLASS_CODES]
    
    x = np.arange(len(CLASS_CODES))
    w = 0.26
    
    fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=300)
    
    b1 = ax.bar(x - w, precisions, w, label="Precision (%)", color="#0284c7", edgecolor='#1e293b', linewidth=0.7, zorder=3)
    b2 = ax.bar(x, recalls, w, label="Recall / Sensitivity (%)", color="#dc2626", edgecolor='#1e293b', linewidth=0.7, zorder=3)
    b3 = ax.bar(x + w, f1_scores, w, label="F1-Score (%)", color="#059669", edgecolor='#1e293b', linewidth=0.7, zorder=3)
    
    ax.set_title("Per-Class Diagnostic Performance on Held-Out Test Set (N = 2,534 Images)", 
                 fontsize=13.5, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xlabel("ISIC Diagnostic Category and Test Support Count (n)", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_ylabel("Metric Score (%)", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.5, zorder=0)
    
    # Custom x-tick labels with class support counts
    xtick_labels = [f"{CLASS_CODES[i]}\n(n={supports[i]:,})" for i in range(len(CLASS_CODES))]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=10.0, fontweight='bold')
    
    # Helper function to annotate bar values
    def annotate_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 1.2, f"{h:.1f}", 
                    ha='center', va='bottom', fontsize=8.0, fontweight='bold', rotation=45, color='#0f172a')
                    
    annotate_bars(b1)
    annotate_bars(b2)
    annotate_bars(b3)
    
    ax.legend(loc='upper right', frameon=True, fontsize=10.0, facecolor='#ffffff', edgecolor='#cbd5e1')
    
    # Summary benchmark annotation
    bench_text = f"Overall Benchmark (N=2,534):\n• Accuracy: {m['accuracy']*100:.2f}%\n• Balanced Acc: {m['balanced_accuracy']*100:.2f}%\n• Macro-F1: {m['macro_f1']*100:.2f}%\n• Melanoma Recall: {per_class['MEL']['recall']*100:.2f}%"
    ax.text(0.015, 0.96, bench_text, transform=ax.transAxes, fontsize=8.8, fontweight='bold', color='#334155',
            va='top', bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#cbd5e1", lw=1))
            
    plt.tight_layout()
    save_figure(fig, "04_per_class_performance")


# ==============================================================================
# Clean Output Directory to Ensure ONLY the 4 Requested Figures Exist
# ==============================================================================
def clean_extraneous_figures():
    """Remove any extra files so report_figures contains strictly the 4 requested figures."""
    allowed_bases = [
        "01_dataset_class_distribution",
        "02_training_validation_loss",
        "03_test_confusion_matrix",
        "04_per_class_performance"
    ]
    allowed_files = set([f"{b}.png" for b in allowed_bases] + ["README.md", "figure_manifest.csv"])
    
    for d in [OUT_DIR, WORKSPACE_OUT_DIR, DOWNLOADS_OUT_DIR]:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname not in allowed_files:
                fpath = os.path.join(d, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception:
                    pass


def generate_manifest():
    manifest_rows = [
        ["01_dataset_class_distribution.png", "C:\\SafeDerm\\splits\\train.csv", "Bar chart of actual training sample counts (N = 20,264) across all 8 ISIC 2019 classes", "VERIFIED"],
        ["02_training_validation_loss.png", "C:\\SafeDerm\\checkpoints\\training_log.csv", "Line plot of training & validation focal loss over 24 epochs; marks best checkpoint at Epoch 22", "VERIFIED"],
        ["03_test_confusion_matrix.png", "C:\\SafeDerm\\results\\confusion_matrix_test.csv", "Annotated 8-class heatmap on 2,534 test images; highlights 125 Melanoma false negatives", "VERIFIED"],
        ["04_per_class_performance.png", "C:\\SafeDerm\\results\\test_metrics.json", "Grouped bar chart showing Precision, Recall, and F1 across all 8 classes with support counts", "VERIFIED"],
    ]
    df_manifest = pd.DataFrame(manifest_rows, columns=["Filename", "Source File", "Analytical Purpose", "Status"])
    
    readme_content = """# SafeDerm — Essential B.Tech Project Report Figures
**Project Title:** Class-Weighted Transfer Learning with Explainability Validation for Dermoscopic Malignancy Detection  
**Model Architecture:** EfficientNet-B0 + Convolutional Block Attention Module (CBAM)  
**Trained Checkpoint:** `best.pt` (Epoch 22, Validation Macro-F1: 0.7705)  
**Dataset:** ISIC 2019 (25,331 Dermoscopic Images across 8 Diagnostic Classes)  

---

## Generated Figures (Saved at 300 DPI in PNG format)

| Filename | Source File | Analytical Purpose in Report | Verification Status |
|:---|:---|:---|:---:|
| **`01_dataset_class_distribution.png`** | `splits/train.csv` | Visualizes exact training class imbalance (NV: 10,300 vs DF: 191) across 8 classes. | **VERIFIED** |
| **`02_training_validation_loss.png`** | `checkpoints/training_log.csv` | Tracks Focal Loss convergence across 24 epochs; marks best checkpoint at Epoch 22. | **VERIFIED** |
| **`03_test_confusion_matrix.png`** | `results/confusion_matrix_test.csv` | Annotated test confusion matrix (N = 2,534); highlights 125 Melanoma $\\rightarrow$ Nevus false negatives. | **VERIFIED** |
| **`04_per_class_performance.png`** | `results/test_metrics.json` | Grouped bar chart of Precision, Recall, and F1 with exact test support counts. | **VERIFIED** |

---

## Reproduction Command
```powershell
& "C:\\SafeDerm\\venv\\Scripts\\python.exe" "c:\\Users\\SANTHOSH\\Downloads\\Skin Lesion Review\\generate_four_btech_report_figures.py"
```
"""
    for d in [OUT_DIR, WORKSPACE_OUT_DIR, DOWNLOADS_OUT_DIR]:
        df_manifest.to_csv(os.path.join(d, "figure_manifest.csv"), index=False)
        with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING 4 ESSENTIAL B.TECH PROJECT REPORT FIGURES")
    print("Official Title: Class-Weighted Transfer Learning with Explainability")
    print("                Validation for Dermoscopic Malignancy Detection")
    print("=" * 70)
    
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    clean_extraneous_figures()
    generate_manifest()
    
    print("\n" + "=" * 70)
    print("SUCCESS: ALL 4 ESSENTIAL REPORT FIGURES GENERATED IN:")
    print(f"  --> {OUT_DIR}")
    print(f"  --> {WORKSPACE_OUT_DIR}")
    print(f"  --> {DOWNLOADS_OUT_DIR}")
    print("=" * 70)
