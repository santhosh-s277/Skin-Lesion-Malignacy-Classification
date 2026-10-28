# SafeDerm — Essential B.Tech Project Report Figures
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
| **`03_test_confusion_matrix.png`** | `results/confusion_matrix_test.csv` | Annotated test confusion matrix (N = 2,534); highlights 125 Melanoma $\rightarrow$ Nevus false negatives. | **VERIFIED** |
| **`04_per_class_performance.png`** | `results/test_metrics.json` | Grouped bar chart of Precision, Recall, and F1 with exact test support counts. | **VERIFIED** |

---

## Reproduction Command
```powershell
& "C:\SafeDerm\venv\Scripts\python.exe" "c:\Users\SANTHOSH\Downloads\Skin Lesion Review\generate_four_btech_report_figures.py"
```
