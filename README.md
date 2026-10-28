# Skin Lesion Malignancy Classification: SafeDerm Triage Lite

> **Explainable Deep Learning for Dermoscopic Skin Lesion Triage**  
> *ISIC 2019 Research Showcase & B.Tech Capstone Project*

---

### ⚠️ Mandatory Medical Disclaimer
> **Research prototype. Not a medical device. Not a clinical diagnosis. Consult a qualified clinician.**  
> *Never use predictions for medical diagnosis, cancer screening, or treatment recommendations without histological biopsy and licensed dermatologist evaluation.*

---

## 1. Executive Summary & Architecture

This repository contains the technical implementation, diagnostic auditing scripts, explainability validation suites, and publication-ready evaluation figures for **Skin Lesion Malignancy Classification** using **SafeDerm Triage Lite**.

The system performs multi-class dermoscopic triage across eight **ISIC 2019** diagnostic categories:

| Index | Code | Full Category Name | Class Type | Clinical Priority |
|:---:|:---:|:---|:---|:---:|
| 0 | **MEL** | Melanoma | Malignant Melanocytic | **Critical** |
| 1 | **NV** | Melanocytic Nevus | Benign Melanocytic (Mole) | Routine |
| 2 | **BCC** | Basal Cell Carcinoma | Malignant Keratinocyte | Urgent |
| 3 | **AK** | Actinic Keratosis / Intraepithelial Carcinoma | Pre-Malignant | Monitoring |
| 4 | **BKL** | Benign Keratosis (Solar Lentigo / Seborrheic / LPLK) | Benign Keratinocyte | Routine |
| 5 | **DF** | Dermatofibroma | Benign Fibrous Histiocytoma | Routine |
| 6 | **VASC** | Vascular Lesion (Hemangioma / Angioma) | Benign Vascular | Routine |
| 7 | **SCC** | Squamous Cell Carcinoma | Malignant Keratinocyte | Urgent |

### Model Specifications
- **Backbone:** TorchVision `EfficientNet-B0` with **CBAM (Convolutional Block Attention Module)** inserted after feature stages 3 (40 ch), 5 (112 ch), and 7 (320 ch).
- **Trained Weights:** `best.pt` (Epoch 22, Val Macro-F1: 0.7705).
- **Optimization:** Class-weighted Focal Loss ($\gamma = 2.0$) addressing severe ISIC 2019 class imbalance.
- **Strict Parameter Breakdown:**
  - Trainable parameters: `4,033,797`
  - Non-trainable BatchNorm buffers: `42,065`
  - Total `state_dict` elements: `4,075,862`

---

## 2. Key Performance Metrics (Held-Out Test Set: N = 2,534)

| Metric | Result | Context |
|:---|:---:|:---|
| **Overall Accuracy** | **82.2%** | Test set evaluation |
| **Balanced Accuracy** | **74.5%** | Class-weighted average |
| **Macro-F1 Score** | **76.1%** | Balanced harmonic mean |
| **Melanoma (MEL) Recall** | **64.2%** | Key focal point for clinical safety review |

---

## 3. Explainability & Grad-CAM Sanity Validation

Grad-CAM computes spatial sensitivity maps targeting feature stage 5 (`model.features[5]`, $14 \times 14$ activation map) with continuous float bilinear interpolation to $224 \times 224$:

$$\alpha_k^c = \frac{1}{HW} \sum_{i=1}^{H}\sum_{j=1}^{W} \frac{\partial y^c}{\partial A_{ij}^k}, \quad L_{\text{GradCAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### 5-Test Diagnostic Sanity Suite:
1. **Forward/Backward Validity:** Confirms finite gradient tensors in $[0, 1]$ and non-zero maps.
2. **Hook Lifecycle & Memory Stability:** Strict `try...finally` cleanup with zero leaked PyTorch hooks or memory growth across calls.
3. **Class Discriminability:** Verifies distinct heatmap activations between competing diagnostic logits (e.g. NV vs MEL vs VASC).
4. **Parameter Randomization Check (Adebayo et al.):** Confirms heatmap decouples when classifier weights are randomized (preventing edge-detector artifacts).
5. **Artifact Stress & Occlusion Faithfulness:** Evaluates resilience against dermoscopic hair artifacts, vignetting rings, and central occlusions.

---

## 4. Repository Structure

```
├── report_figures/                                     # Publication & report figures (300 DPI)
│   ├── 01_dataset_class_distribution.png              # Class distribution & imbalance
│   ├── 02_training_validation_loss.png                # Focal Loss convergence across epochs
│   ├── 03_test_confusion_matrix.png                   # Annotated confusion matrix (N=2,534)
│   ├── 04_per_class_performance.png                   # Precision, Recall & F1 per category
│   ├── figure_manifest.csv                            # Figure metadata manifest
│   └── README.md                                      # Documentation for report figures
├── extracted_images/                                   # Extracted dermoscopic test sample figures
├── Skin_Lesion_Malignancy_Classification_BTech_Project_Audit.md   # Complete technical project audit
├── Skin_Lesion_Malignancy_Classification_BTech_Project_Audit.html # HTML version of audit report
├── build_app.py                                        # Gradio dual-tab UI builder
├── build_gradcam.py                                    # Mathematical Grad-CAM implementation
├── build_modules.py                                    # CBAM, backbone & preprocessing modules
├── build_validate_gradcam.py                           # 5-test Grad-CAM validation suite
├── run_full_gradcam_diagnostics.py                     # Full diagnostic runner & visualizer
├── generate_four_btech_report_figures.py               # Generates 300 DPI publication figures
├── test_app_integration.py                             # Automated integration tests
├── test_core.py                                        # Core inference verification tests
├── splits-20261013T162901Z-1-001.zip                   # Dataset splits archive
└── .gitignore                                          # Git ignore configuration
```

---

## 5. Usage & Reproduction

### Prerequisites
- Python 3.10 or 3.11
- PyTorch with CUDA support (or CPU fallback)
- Gradio, Pillow, NumPy, Matplotlib, Seaborn

### Running the Validation Suite:
```powershell
python build_validate_gradcam.py
```

### Running Full Diagnostics:
```powershell
python run_full_gradcam_diagnostics.py
```

### Generating Report Figures:
```powershell
python generate_four_btech_report_figures.py
```

---

## 6. Authors & Academic Credits

- **Department:** Artificial Intelligence and Machine Learning
- **Lead Candidate:** SANTHOSH S (`7376242AL191`)
- **Team Members:** NISHANTH R (`7376242AL161`), SANJAY K (`7376242AL187`)
- **Institution:** Bannari Amman Institute of Technology
