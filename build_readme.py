# Generator for C:\SafeDerm\README.md
# Comprehensive documentation for SafeDerm Triage Lite

import os

readme_path = r"C:\SafeDerm\README.md"

readme_content = """# SafeDerm Triage Lite
> **Explainable AI for Educational Skin Lesion Image Classification**  
> *ISIC 2019 College Research Showcase*

---

### ⚠️ Mandatory Medical Disclaimer
> **Research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.**  
> *Never describe predictions as a medical diagnosis. Do not use for cancer screening or treatment recommendations.*

---

## 1. Executive Summary & Architecture

SafeDerm Triage Lite demonstrates explainable deep learning for dermoscopic image triage across eight **ISIC 2019** diagnostic categories:

| Index | Code | Full Category Name | Class Type |
|:---:|:---:|:---|:---|
| 0 | **MEL** | Melanoma | Malignant Melanocytic |
| 1 | **NV** | Melanocytic Nevus | Benign Melanocytic (Mole) |
| 2 | **BCC** | Basal Cell Carcinoma | Malignant Keratinocyte |
| 3 | **AK** | Actinic Keratosis / Intraepithelial Carcinoma | Pre-Malignant |
| 4 | **BKL** | Benign Keratosis (Solar Lentigo / Seborrheic / LPLK) | Benign Keratinocyte |
| 5 | **DF** | Dermatofibroma | Benign Fibrous Histiocytoma |
| 6 | **VASC** | Vascular Lesion (Hemangioma / Angioma) | Benign Vascular |
| 7 | **SCC** | Squamous Cell Carcinoma | Malignant Keratinocyte |

### Model Specifications
- **Backbone:** TorchVision `EfficientNet-B0` with **CBAM (Convolutional Block Attention Module)** inserted after feature stages 3 (40 ch), 5 (112 ch), and 7 (320 ch).
- **Trained Weights:** `C:\\SafeDerm\\checkpoints\\best.pt` (Epoch 22, Val Macro-F1: 0.7705).
- **Strict Parameter Breakdown:**
  - Trainable parameters: `4,033,797`
  - Non-trainable BatchNorm buffers (`running_mean`, `running_var`, `num_batches_tracked`): `42,065`
  - Total `state_dict` elements: `4,075,862` (Strict load: 0 missing, 0 unexpected keys).
- **Hardware Acceleration:** Native CUDA support on NVIDIA GeForce RTX 4050 Laptop GPU (~6 GB VRAM) with automatic CPU fallback.

---

## 2. Preprocessing & Input Validation

### Documented Preprocessing Policy
All images undergo exact ISIC evaluation preprocessing before tensor ingestion:
1. **Resizing:** Bicubic interpolation to $256 \\times 256$ pixels.
2. **Center Crop:** Center crop to $224 \\times 224$ pixels.
3. **Tensor Conversion:** Conversion to floating-point tensor in $[0, 1]$.
4. **ImageNet Normalization:** 
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`

### Server-Side Validation Pipeline
The application enforces strict server-side validation against:
- Empty or zero-byte uploads
- File sizes exceeding **10 MB**
- Corrupt, unreadable, or truncated image files (safe Pillow verify)
- Minimum dimension threshold (< $64 \\times 64$ pixels)
- Maximum decoded pixel limit (> 20 Megapixels)
- Unsupported file formats (supported: JPG, JPEG, PNG, WebP)
- CUDA out-of-memory errors (automatic cache flush & recovery)

---

## 3. Dual UI User Experience

The application presents two complete interfaces within a unified Gradio application:

### UI A: Image Analysis Studio
- **Dermoscopic Upload Card:** Drag-and-drop or clipboard upload with auto-preview and form reset.
- **Top Class Result Card:** Distinct, color-coded badges for malignant (`MEL`), carcinoma/pre-malignant (`BCC`, `SCC`, `AK`), and benign (`NV`, `BKL`, `DF`, `VASC`) predictions with uncalibrated softmax scores.
- **Top-3 Ranked Classes:** Ranked horizontal progress bars with exact model scores.
- **3-Panel Grad-CAM Explainability Grid:**
  1. *Preprocessed Input:* Exact un-normalized $224 \\times 224$ RGB image presented to the model.
  2. *Sensitivity Heatmap:* Normalized class activation map colorized with Jet colormap.
  3. *Blended Overlay:* 50% alpha blend highlighting spatial model attention.
  4. *Relative Activation Scale:* Colorbar legend ranging from Low Sensitivity (0.0) to High Sensitivity (1.0).
- **Complete 8-Class Breakdown:** Expandable accordion displaying raw softmax outputs for all 8 ISIC categories.

### UI B: Model Insights & Performance Dashboard
- **4 Baseline KPI Metric Cards:** Loaded directly from frozen evaluation artifact `results/test_metrics.json` on 2,534 held-out test images:
  - Overall Test Accuracy: **82.2%**
  - Balanced Accuracy: **74.5%**
  - Macro-F1 Score: **76.1%**
  - Melanoma Recall: **64.2%** *(Highlighted as primary clinical limitation)*
- **Held-Out Test Set Confusion Matrix:** High-resolution normalized matrix visualization.
- **Per-Class Recall & F1-Score Breakdown:** Comparative bar chart highlighting performance across common vs rare classes.
- **Architecture Explorer:** Interactive flow diagram of the EfficientNet-B0 + CBAM pipeline.
- **Dataset Limitations Panel:** Detailed explanation of class imbalance, absence of external cohort validation, and shortcut learning hazards.

---

## 4. Grad-CAM Formulation & Sanity Suite

### Mathematical Formulation
Grad-CAM computes the spatial sensitivity map with respect to the raw target class logit $y^c$:

$$\\alpha_k^c = \\frac{1}{HW} \\sum_{i=1}^{H}\\sum_{j=1}^{W} \\frac{\\partial y^c}{\\partial A_{ij}^k}$$

$$L_{\\text{GradCAM}}^c = \\text{ReLU}\\left( \\sum_k \\alpha_k^c A^k \\right)$$

- **Target Layer:** `model.features[5]` (Stage 5 InvertedResidual + CBAM, output shape: `[1, 112, 14, 14]`).
- **Continuous Float Interpolation:** Bilinear tensor interpolation (`F.interpolate`) directly to $224 \\times 224$, avoiding 8-bit quantization noise and bicubic edge ringing.
- **Hook Lifecycle:** `try/finally` block guarantees hook removal and zero memory leaks.

### Diagnostic Sanity Suite (`validate_gradcam.py`)
Run the comprehensive 5-test validation suite:
```powershell
C:\\SafeDerm\\venv\\Scripts\\python.exe C:\\SafeDerm\\src\\validate_gradcam.py
```

| Test | Objective | Measured Result | Verdict |
|:---|:---|:---|:---:|
| **Test A: Forward/Backward Validity** | Verify activation shape, gradients, finite numbers, and zero-map handling | Activation `[1, 112, 14, 14]`, CAM `[224, 224]`, finite in `[0, 1]` | **PASS** |
| **Test B: Stability & Memory** | Check hook cleanup, reproducibility, and CUDA memory across 5 calls | 0 leaked hooks, max diff `< 1e-5`, memory delta = 0 bytes | **PASS** |
| **Test C: Class Discriminability** | Ensure heatmap varies for different target class logits | NV vs MEL diff = `0.1322`, NV vs VASC diff = `0.1701`, weight corr = `-0.082` | **PASS** |
| **Test D: Parameter Randomization** | Adebayo et al. (2018) sanity check: randomize classifier weights | MAD = `0.1002`, Spatial correlation = `0.0010` (decouples from weights) | **PASS** |
| **Test E: Artifact Stress Test** | Measure sensitivity to circular vignetting and center occlusion | Original (VASC: 43.9%), Vignetted (BCC: 41.8%), Occluded (BCC: 39.2%) | **PASS** |

---

## 5. Quick Launch Guide (Windows PowerShell)

### Launch the Application Locally:
```powershell
C:\\SafeDerm\\venv\\Scripts\\python.exe C:\\SafeDerm\\src\\app.py
```

Open your browser to:
**`http://127.0.0.1:7860`**

*(Public sharing is disabled for security and research integrity.)*

### Run Integration Tests:
```powershell
C:\\SafeDerm\\venv\\Scripts\\python.exe "c:\\Users\\SANTHOSH\\Downloads\\Skin Lesion Review\\test_app_integration.py"
```

---

## 6. College Presentation Screenshot Guidance

When presenting to evaluators, capture these 5 key screenshots:
1. **Studio Prediction Overview:** Upload `sample_test_image.jpg`, click *Analyze Image*, and capture the top class card (`NV — Melanocytic Nevus (93.9%)`) and Top-3 ranked progress bars.
2. **Explainability 3-Panel Grid:** Capture the side-by-side Input, Sensitivity Map (14×14), Blended Overlay, and Relative Activation Legend.
3. **8-Class Breakdown:** Expand the raw softmax table showing complete probabilities across all 8 ISIC categories.
4. **Performance Dashboard (UI B):** Capture the 4 KPI cards (82.2% Accuracy, 74.5% Balanced Accuracy, 76.1% Macro-F1, 64.2% Melanoma Recall) alongside the Confusion Matrix.
5. **Architecture Explorer:** Capture the workflow diagram showing EfficientNet-B0 stages, CBAM attention, and parameter count verification.

---

### Final Mandatory Clinical Notice
> **Research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.**
"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"Updated {readme_path}")
