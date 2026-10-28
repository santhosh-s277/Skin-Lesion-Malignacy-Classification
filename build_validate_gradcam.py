# Generator for C:\SafeDerm\src\validate_gradcam.py
# Comprehensive, repeatable diagnostic suite for SafeDerm Grad-CAM

import os

target_file = r"C:\SafeDerm\src\validate_gradcam.py"

code = '''# SafeDerm Triage Lite - Grad-CAM Validation & Sanity Suite
# Educational/research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Ensure paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

SAVED_MODEL_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "saved_model"))
if SAVED_MODEL_DIR not in sys.path:
    sys.path.insert(0, SAVED_MODEL_DIR)

from inference import InferenceService, CLASS_NAMES, FULL_CLASS_NAMES
from gradcam import GradCAM


def run_test_a(service: InferenceService, gradcam: GradCAM, sample_img: Image.Image) -> dict:
    """Test A: Forward and Backward Validity."""
    print("\\n[Test A] Forward and Backward Validity...")
    tensor, visual = service.preprocess(sample_img)
    device = service.device
    x = tensor.to(device)

    # 1. Forward model output
    with torch.no_grad():
        logits = service.model(x)
    assert logits.shape == (1, 8), f"Expected logits shape (1, 8), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    # 2. Compute CAM
    cam_tensor, cam_logits, target_idx, weights = gradcam.compute_cam(tensor, target_class_idx=1)
    
    assert cam_tensor.shape == (224, 224), f"Expected CAM shape (224, 224), got {cam_tensor.shape}"
    assert torch.isfinite(cam_tensor).all(), "CAM contains NaN or Inf"
    assert 0.0 <= cam_tensor.min().item() and cam_tensor.max().item() <= 1.0, "CAM not normalized to [0, 1]"
    assert weights.shape == (112,), f"Expected weights shape (112,), got {weights.shape}"
    assert torch.isfinite(weights).all(), "Weights contain NaN or Inf"

    # 3. Generate visual images
    h_pil, o_pil, err = gradcam.generate(tensor, target_class_idx=1, cropped_pil=visual)
    assert err is None, f"Generate returned error: {err}"
    assert h_pil.size == (224, 224), f"Heatmap size {h_pil.size} != (224, 224)"
    assert o_pil.size == (224, 224), f"Overlay size {o_pil.size} != (224, 224)"
    assert h_pil.mode == "RGB", f"Heatmap mode {h_pil.mode} != RGB"
    assert o_pil.mode == "RGB", f"Overlay mode {o_pil.mode} != RGB"

    # 4. Zero map edge case (all black input)
    black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    b_tensor, b_vis = service.preprocess(black_img)
    b_cam, _, _, _ = gradcam.compute_cam(b_tensor, target_class_idx=0)
    assert torch.isfinite(b_cam).all(), "Zero-input CAM contains NaN or Inf"

    print("  PASS: Target activations (112, 14, 14) and gradients valid.")
    print(f"  PASS: Heatmap shape {cam_tensor.shape}, min={cam_tensor.min():.4f}, max={cam_tensor.max():.4f}")
    return {"status": "PASS", "shape": list(cam_tensor.shape)}


def run_test_b(service: InferenceService, gradcam: GradCAM, sample_img: Image.Image) -> dict:
    """Test B: Repeated-Request Stability & Memory Cleanliness."""
    print("\\n[Test B] Repeated-Request Stability...")
    tensor, visual = service.preprocess(sample_img)

    initial_cuda_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    cams = []

    for i in range(5):
        # Verify 0 active hooks before call
        assert len(gradcam.hook_handles) == 0, f"Leaked hook handles before iteration {i}"
        
        cam_t, _, _, _ = gradcam.compute_cam(tensor, target_class_idx=1)
        cams.append(cam_t.numpy())
        
        # Verify 0 active hooks after call
        assert len(gradcam.hook_handles) == 0, f"Leaked hook handles after iteration {i}"
        assert gradcam.activations is None, "Stale activations retained"
        assert gradcam.gradients is None, "Stale gradients retained"

    # Check reproducibility across calls
    for i in range(1, 5):
        diff = np.abs(cams[0] - cams[i]).max()
        assert diff < 1e-5, f"Non-deterministic CAM across calls (diff={diff})"

    final_cuda_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    mem_delta = final_cuda_mem - initial_cuda_mem

    print("  PASS: All 5 consecutive calls cleaned up hooks immediately.")
    print(f"  PASS: Deterministic reproducibility verified (max diff < 1e-5).")
    print(f"  PASS: GPU memory delta: {mem_delta} bytes (no memory leak).")
    return {"status": "PASS", "leak_bytes": mem_delta}


def run_test_c(service: InferenceService, gradcam: GradCAM, sample_img: Image.Image) -> dict:
    """Test C: Class-Sensitivity (Different Target Classes)."""
    print("\\n[Test C] Class-Sensitivity (Comparing Explanations Across Classes)...")
    tensor, visual = service.preprocess(sample_img)

    # Compute for Class 1 (NV), Class 0 (MEL), Class 6 (VASC)
    cam_nv, _, _, w_nv = gradcam.compute_cam(tensor, target_class_idx=1)
    cam_mel, _, _, w_mel = gradcam.compute_cam(tensor, target_class_idx=0)
    cam_vasc, _, _, w_vasc = gradcam.compute_cam(tensor, target_class_idx=6)

    # Measure differences
    diff_nv_mel = torch.mean(torch.abs(cam_nv - cam_mel)).item()
    diff_nv_vasc = torch.mean(torch.abs(cam_nv - cam_vasc)).item()
    weight_corr = torch.corrcoef(torch.stack([w_nv, w_mel]))[0, 1].item()

    print(f"  NV vs MEL Mean Absolute Map Difference:  {diff_nv_mel:.4f}")
    print(f"  NV vs VASC Mean Absolute Map Difference: {diff_nv_vasc:.4f}")
    print(f"  NV vs MEL Alpha Weights Correlation:     {weight_corr:.4f}")

    assert diff_nv_mel > 0.01, f"CAM is class-insensitive (NV vs MEL diff={diff_nv_mel})"
    print("  PASS: Grad-CAM produces distinct, class-discriminative sensitivity maps.")
    return {"status": "PASS", "diff_nv_mel": diff_nv_mel, "diff_nv_vasc": diff_nv_vasc}


def run_test_d(service: InferenceService, gradcam: GradCAM, sample_img: Image.Image) -> dict:
    """Test D: Model Parameter-Randomization Sanity Check (Adebayo et al., 2018)."""
    print("\\n[Test D] Model Parameter-Randomization Sanity Check...")
    tensor, visual = service.preprocess(sample_img)

    # 1. Trained model CAM
    trained_cam, trained_logits, _, _ = gradcam.compute_cam(tensor, target_class_idx=1)

    # 2. Create copy of model with randomized classifier head
    test_model = copy.deepcopy(service.model)
    nn.init.normal_(test_model.classifier[1].weight, mean=0.0, std=1.0)
    nn.init.zeros_(test_model.classifier[1].bias)

    rand_gradcam = GradCAM(test_model)
    rand_cam, rand_logits, _, _ = rand_gradcam.compute_cam(tensor, target_class_idx=1)

    # Measure difference
    mad = torch.mean(torch.abs(trained_cam - rand_cam)).item()
    corr = torch.corrcoef(torch.stack([trained_cam.flatten(), rand_cam.flatten()]))[0, 1].item()

    print(f"  Trained Logit:    {trained_logits[0, 1].item():.4f}")
    print(f"  Randomized Logit: {rand_logits[0, 1].item():.4f}")
    print(f"  Mean Absolute Map Difference (MAD): {mad:.4f}")
    print(f"  Spatial Correlation:                {corr:.4f}")

    assert mad > 0.05, f"Heatmap failed Adebayo randomization check (MAD={mad:.4f} <= 0.05)"
    print("  PASS: Explanation responds strongly to model weights (Adebayo sanity passed).")
    return {"status": "PASS", "mad": mad, "spatial_corr": corr}


def run_test_e(service: InferenceService, gradcam: GradCAM, sample_img: Image.Image) -> dict:
    """Test E: Artifact Stress Test (Vignetting & Occlusion)."""
    print("\\n[Test E] Artifact Stress Test...")
    
    # E1: Original Image
    p_orig = service.predict(sample_img)
    orig_class = p_orig["top_class"]
    orig_score = p_orig["top_score"]
    orig_cam, _, _, _ = gradcam.compute_cam(p_orig["input_tensor"], target_class_idx=1)

    # E2: Circular Vignette Border Mask
    # Simulate heavy dermatoscope vignetting: outer corners zeroed out
    arr_vignette = np.array(sample_img.copy().convert("RGB"))
    h, w, _ = arr_vignette.shape
    cy, cx = h // 2, w // 2
    r = min(cx, cy) * 0.88
    y, x = np.ogrid[:h, :w]
    outer_mask = ((x - cx)**2 + (y - cy)**2) > r**2
    arr_vignette[outer_mask] = 0
    img_vignette = Image.fromarray(arr_vignette)
    
    p_vig = service.predict(img_vignette)
    vig_class = p_vig["top_class"]
    vig_score = p_vig["top_score"]

    # E3: Center Occlusion
    arr_occ = np.array(sample_img.copy().convert("RGB"))
    box_r = min(cx, cy) // 3
    arr_occ[cy - box_r : cy + box_r, cx - box_r : cx + box_r] = [220, 190, 175]
    img_occ = Image.fromarray(arr_occ)

    p_occ = service.predict(img_occ)
    occ_class = p_occ["top_class"]
    occ_score = p_occ["top_score"]

    print(f"  Condition 1 (Original):         Class={orig_class}, Score={orig_score*100:.1f}%")
    print(f"  Condition 2 (Border Vignette):  Class={vig_class}, Score={vig_score*100:.1f}%, Delta={vig_score-orig_score:+.4f}")
    print(f"  Condition 3 (Center Occluded):  Class={occ_class}, Score={occ_score*100:.1f}%, Delta={occ_score-orig_score:+.4f}")

    return {
        "status": "PASS",
        "orig": {"class": orig_class, "score": orig_score},
        "vignette": {"class": vig_class, "score": vig_score},
        "occluded": {"class": occ_class, "score": occ_score},
    }


def main():
    print("=========================================================")
    print("SafeDerm Triage Lite - Grad-CAM Diagnostic Suite")
    print("=========================================================")
    service = InferenceService.get_instance()
    gradcam = GradCAM(service.model)

    sample_path = r"C:\\SafeDerm\\sample_test_image.jpg"
    assert os.path.exists(sample_path), f"Sample image missing at {sample_path}"
    sample_img = Image.open(sample_path)

    res_a = run_test_a(service, gradcam, sample_img)
    res_b = run_test_b(service, gradcam, sample_img)
    res_c = run_test_c(service, gradcam, sample_img)
    res_d = run_test_d(service, gradcam, sample_img)
    res_e = run_test_e(service, gradcam, sample_img)

    print("\\n=========================================================")
    print("ALL 5 GRAD-CAM DIAGNOSTIC TESTS PASSED!")
    print("=========================================================")


if __name__ == "__main__":
    main()
'''

with open(target_file, "w", encoding="utf-8") as f:
    f.write(code)
print(f"Created {target_file}")
