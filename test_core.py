# Test suite for SafeDerm Triage Lite core modules
import sys
import os
import torch
from PIL import Image

sys.path.insert(0, r"C:\SafeDerm\src")
sys.path.insert(0, r"C:\SafeDerm\saved_model")

from inference import InferenceService, CLASS_NAMES, FULL_CLASS_NAMES
from gradcam import GradCAM
import dashboard

print("=" * 60)
print("RUNNING SAFEDERM CORE MODULE TESTS")
print("=" * 60)

# Test 1: Inference Service Initialization
service = InferenceService.get_instance()
print("[Test 1 PASS] InferenceService initialized.")
print(f"  Device: {service.device}")
print(f"  Epoch: {service.epoch}")
print(f"  Val Macro-F1: {service.val_macro_f1:.4f}")
print(f"  Classes: {CLASS_NAMES}")

# Test 2: Synthetic inference test
dummy_img = Image.new("RGB", (256, 256), color=(200, 150, 120))
res = service.predict(dummy_img)
assert res["success"] is True, f"Prediction failed: {res['error']}"
print("[Test 2 PASS] Inference smoke test succeeded.")
print(f"  Highest scoring class: {res['top_class']} ({res['top_full_name']})")
print(f"  Top score: {res['top_score']:.4f}")
print("  Top 3:", [(item['class'], item['percentage']) for item in res['top3']])
assert len(res["scores"]) == 8, f"Expected 8 class scores, got {len(res['scores'])}"

# Test 3: Grad-CAM generation
cam = GradCAM(service.model)
top_idx = CLASS_NAMES.index(res["top_class"])
heatmap, overlay, cam_err = cam.generate(
    input_tensor=res["input_tensor"],
    target_class_idx=top_idx,
    cropped_pil=res["cropped_img"]
)
assert cam_err is None, f"Grad-CAM error: {cam_err}"
assert heatmap is not None and overlay is not None, "Grad-CAM outputs are None"
assert overlay.size == (224, 224), f"Expected 224x224 overlay, got {overlay.size}"
print("[Test 3 PASS] Grad-CAM generation succeeded on features[5] (14x14 layer).")
print(f"  Heatmap size: {heatmap.size}, mode: {heatmap.mode}")
print(f"  Overlay size: {overlay.size}, mode: {overlay.mode}")

# Test 4: Dashboard artifacts
metrics = dashboard.load_test_metrics()
assert "accuracy" in metrics, "Missing accuracy in test_metrics"
assert "macro_f1" in metrics, "Missing macro_f1 in test_metrics"
cm_img = dashboard.get_confusion_matrix_image()
assert cm_img is not None, "Failed to load confusion matrix image"
perf_img = dashboard.get_class_performance_plot()
assert perf_img is not None, "Failed to generate class performance plot"
arch_svg = dashboard.get_architecture_svg()
assert len(arch_svg) > 100, "Architecture SVG too small"
print("[Test 4 PASS] Dashboard artifact loaders succeeded.")
print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
print(f"  Test Macro-F1: {metrics['macro_f1']:.4f}")
print(f"  Melanoma Recall: {metrics['per_class']['MEL']['recall']:.4f}")

# Test 5: Validation edge cases
# Corrupt input
_, err_none = service.validate_image_input(None)
assert err_none is not None, "Expected error on None input"

# Too small image
small_img = Image.new("RGB", (32, 32))
_, err_small = service.validate_image_input(small_img)
assert "too small" in err_small.lower(), f"Unexpected error message for small image: {err_small}"

# Grayscale conversion
gray_img = Image.new("L", (100, 100), color=128)
valid_rgb, err_gray = service.validate_image_input(gray_img)
assert err_gray is None and valid_rgb.mode == "RGB", "Failed to convert grayscale to RGB"

# RGBA conversion
rgba_img = Image.new("RGBA", (100, 100), color=(100, 100, 100, 255))
valid_rgb, err_rgba = service.validate_image_input(rgba_img)
assert err_rgba is None and valid_rgb.mode == "RGB", "Failed to convert RGBA to RGB"
print("[Test 5 PASS] Input validation edge cases passed.")

print("=" * 60)
print("ALL 5 TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
