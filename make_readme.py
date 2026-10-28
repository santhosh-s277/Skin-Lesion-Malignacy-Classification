import os

readme = """# SafeDerm Triage Lite

> **Disclaimer:** Research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.

## Project Overview
SafeDerm Triage Lite is an educational/research ML prototype for 8-class skin lesion triage based on ISIC 2019:
`MEL` (Melanoma), `NV` (Melanocytic nevus), `BCC` (Basal cell carcinoma), `AK` (Actinic keratosis), `BKL` (Benign keratosis), `DF` (Dermatofibroma), `VASC` (Vascular lesion), `SCC` (Squamous cell carcinoma).

## Structure
- `src/`: Core implementation scripts
- `checkpoints/`: Model checkpoint (`best.pt`, 375 tensors, EfficientNet-B0 + CBAM stages 3, 5, 7)
- `saved_model/`: Model definition and config
- `splits/`: Dataset splits (`val_select.csv`, `val_calib.csv`, `test.csv`)
- `results/lite/`: Calibrated metrics, conformal bounds, and frozen evaluation outputs
- `requirements.txt`: Python package dependencies

## Environment Setup
Run within the dedicated Python 3.11 virtual environment:
```powershell
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```
"""

target = r"C:\SafeDerm\README.md"
with open(target, "w", encoding="utf-8") as f:
    f.write(readme.strip() + "\n")
print(f"Created {target}")
