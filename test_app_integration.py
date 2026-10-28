# test_app_integration.py
# End-to-end integration test for SafeDerm Triage Lite app logic

import os
import sys
import numpy as np
from PIL import Image

SRC_DIR = r"C:\SafeDerm\src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import app
import dashboard
from inference import CLASS_NAMES

def run_tests():
    print("==================================================")
    print("SafeDerm Triage Lite - App Integration Test Suite")
    print("==================================================")
    all_passed = True

    # Test 1: None input
    print("\n[Test 1] analyze_image with None input...")
    out = app.analyze_image(None)
    assert len(out) == 7, f"Expected 7 outputs, got {len(out)}"
    assert "Input Required" in out[0], "Expected error message in summary HTML"
    assert out[3] is None and out[4] is None and out[5] is None, "Expected None images for error"
    print("PASS: Handled None input safely.")

    # Test 2: Too small image (< 64x64)
    print("\n[Test 2] analyze_image with too small image (32x32)...")
    tiny_img = Image.new("RGB", (32, 32), color=(120, 80, 60))
    out = app.analyze_image(tiny_img)
    assert "Validation Notice" in out[0] or "too small" in out[0].lower(), f"Expected validation notice, got {out[0]}"
    assert out[3] is None, "Expected None images on validation error"
    print("PASS: Rejected image smaller than 64x64.")

    # Test 3: Valid synthetic image (300x300 RGB)
    print("\n[Test 3] analyze_image with valid 300x300 RGB image...")
    # Create realistic synthetic lesion (warm brownish center on light skin background)
    arr = np.full((300, 300, 3), [220, 190, 175], dtype=np.uint8)
    # Brown circle
    y, x = np.ogrid[:300, :300]
    mask = (x - 150)**2 + (y - 150)**2 < 70**2
    arr[mask] = [90, 45, 30]
    valid_pil = Image.fromarray(arr)

    out = app.analyze_image(valid_pil)
    summary_html, bars_html, df_scores, cam_orig, cam_heat, cam_over, status = out
    
    assert "Highest-Scoring Model Class" in summary_html, "Missing top class in summary"
    assert "Top-3 Ranked Classes" in bars_html, "Missing top-3 bars"
    assert df_scores is not None and len(df_scores) == 8, f"Expected 8 rows in score table, got {len(df_scores)}"
    assert isinstance(cam_orig, Image.Image), f"Expected PIL image for cam_orig, got {type(cam_orig)}"
    assert isinstance(cam_heat, Image.Image), f"Expected PIL image for cam_heat, got {type(cam_heat)}"
    assert isinstance(cam_over, Image.Image), f"Expected PIL image for cam_over, got {type(cam_over)}"
    assert cam_orig.size == (224, 224), f"Expected 224x224 cam_orig, got {cam_orig.size}"
    assert cam_heat.size == (224, 224), f"Expected 224x224 cam_heat, got {cam_heat.size}"
    assert cam_over.size == (224, 224), f"Expected 224x224 cam_over, got {cam_over.size}"
    assert "Inference complete" in status, f"Unexpected status: {status}"
    print(f"PASS: Live inference succeeded. Status: {status}")
    print(f"      Top-3 scores rendered in HTML: {len(bars_html)} chars.")
    print(f"      Grad-CAM visualizations generated: {cam_orig.size}")

    # Test 4: clear_analysis
    print("\n[Test 4] clear_analysis reset function...")
    out_clear = app.clear_analysis()
    assert len(out_clear) == 7
    assert "No Image Analyzed" in out_clear[0]
    assert out_clear[1] == ""
    assert out_clear[2] is None
    print("PASS: Form reset returns initial clean state.")

    # Test 5: Dashboard renderers
    print("\n[Test 5] Dashboard components...")
    cm_img = dashboard.get_confusion_matrix_image()
    assert isinstance(cm_img, Image.Image), f"Expected PIL Image for CM, got {type(cm_img)}"
    print(f"PASS: Confusion matrix rendered (size: {cm_img.size})")

    per_class_img = dashboard.get_class_performance_plot()
    assert isinstance(per_class_img, Image.Image), f"Expected PIL Image for per-class chart, got {type(per_class_img)}"
    print(f"PASS: Per-class chart rendered (size: {per_class_img.size})")

    svg_code = dashboard.get_architecture_svg()
    assert "<div" in svg_code and "</div>" in svg_code, "Invalid architecture HTML"
    assert "EfficientNet-B0" in svg_code, "Missing backbone in architecture"
    print(f"PASS: Architecture HTML/diagram rendered ({len(svg_code)} chars)")

    print("\n==================================================")
    print("ALL APP INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
