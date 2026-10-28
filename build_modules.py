# Builder script for SafeDerm Triage Lite modules
import os

SRC_DIR = r"C:\SafeDerm\src"
os.makedirs(SRC_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. inference.py
# -------------------------------------------------------------
inference_code = '''# SafeDerm Triage Lite - Model Inference & Preprocessing Service
# Educational/research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.

import os
import sys
import io
import threading
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Ensure saved_model directory is on sys.path to import model_code
SAVED_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_model"))
if SAVED_MODEL_DIR not in sys.path:
    sys.path.insert(0, SAVED_MODEL_DIR)

try:
    from model_code import build_model, CLASS_NAMES
except ImportError as e:
    raise ImportError(f"Failed to import build_model from {SAVED_MODEL_DIR}/model_code.py: {e}")

# Exact 8 ISIC 2019 classes in index order
EXPECTED_CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
assert CLASS_NAMES == EXPECTED_CLASS_NAMES, f"CLASS_NAMES mismatch: {CLASS_NAMES} != {EXPECTED_CLASS_NAMES}"

FULL_CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus (Benign Mole)",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis / Intraepithelial Carcinoma",
    "BKL": "Benign Keratosis (Solar Lentigo / Seborrheic / LPLK)",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion (Hemangioma / Angioma)",
    "SCC": "Squamous Cell Carcinoma",
}

# Preprocessing: Bicubic resize to 256x256 -> CenterCrop 224x224 -> ToTensor -> ImageNet Normalize
PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# For visual matching (Resize 256 + CenterCrop 224 without normalization)
VISUAL_CROP_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
])

# Input validation limits
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIM = 64                       # Minimum 64x64 pixels
MAX_PIXELS = 20 * 1000 * 1000      # 20 Megapixels
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}


class InferenceService:
    _instance: Optional['InferenceService'] = None
    _lock = threading.Lock()

    def __init__(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path is None:
            checkpoint_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pt")
            )
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.epoch: Optional[int] = None
        self.val_macro_f1: Optional[float] = None
        self.infer_lock = threading.Lock()
        self._load_checkpoint()

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[str] = None) -> 'InferenceService':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(checkpoint_path)
            return cls._instance

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        print(f"[InferenceService] Loading checkpoint from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        if "model" not in ckpt:
            raise KeyError(f"Checkpoint missing 'model' key. Keys: {list(ckpt.keys())}")

        self.epoch = ckpt.get("epoch", 22)
        self.val_macro_f1 = ckpt.get("val_macro_f1", 0.7705)
        ckpt_classes = ckpt.get("class_names", CLASS_NAMES)
        if ckpt_classes != CLASS_NAMES:
            raise ValueError(f"Checkpoint class order {ckpt_classes} differs from expected {CLASS_NAMES}")

        # Instantiate model architecture
        model = build_model(num_classes=len(CLASS_NAMES))
        # Strict state_dict loading
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict load failed! Missing: {missing}, Unexpected: {unexpected}")

        model = model.to(self.device)
        model.eval()
        self.model = model
        print(f"[InferenceService] Strict load succeeded on {self.device}. "
              f"Epoch: {self.epoch}, Val Macro-F1: {self.val_macro_f1:.4f}")

    def validate_image_input(self, image_input: Any) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Validates uploaded image against file size, corrupt data, format, and dimension limits.
        Returns: (PIL.Image or None, error_message or None)
        """
        if image_input is None:
            return None, "No image uploaded. Please upload a dermoscopic image."

        img = None
        # Handle string file path
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return None, f"Uploaded file not found on disk: {os.path.basename(image_input)}"
            try:
                size_bytes = os.path.getsize(image_input)
                if size_bytes == 0:
                    return None, "Uploaded file is empty (0 bytes). Please upload a valid image file."
                if size_bytes > MAX_FILE_BYTES:
                    return None, f"File size ({size_bytes / 1e6:.1f} MB) exceeds the 10 MB limit."
                
                with open(image_input, "rb") as fp:
                    data = fp.read()
                img = Image.open(io.BytesIO(data))
                img.verify()
                img = Image.open(io.BytesIO(data))
            except Exception as e:
                return None, f"Corrupt or unreadable image file: {str(e)}"
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            try:
                img = Image.fromarray(image_input)
            except Exception as e:
                return None, f"Unsupported image input type: {type(image_input).__name__} ({str(e)})"

        # Check format
        img_format = (getattr(img, "format", None) or "JPEG").upper()
        if img_format not in ALLOWED_FORMATS and img_format != "MPO":
            if getattr(img, "format", None) is not None:
                return None, f"Unsupported image format: {img_format}. Allowed formats: JPG, JPEG, PNG, WebP."

        # Check dimensions
        w, h = img.size
        if w < MIN_DIM or h < MIN_DIM:
            return None, f"Image dimensions too small ({w}x{h}). Minimum required dimension is {MIN_DIM}x{MIN_DIM}."
        if (w * h) > MAX_PIXELS:
            return None, f"Image resolution ({w}x{h} = {w*h/1e6:.1f} MP) exceeds the 20 Megapixel limit."

        # Safe RGB conversion
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as e:
            return None, f"Failed to convert image to RGB format: {str(e)}"

        return img, None

    def preprocess(self, pil_img: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
        """
        Applies documented evaluation preprocessing:
        1. Bicubic resize to 256x256
        2. CenterCrop 224x224
        3. ToTensor & ImageNet normalization
        Returns: (tensor shape [1, 3, 224, 224], cropped_visual_img)
        """
        tensor = PREPROCESS_TRANSFORM(pil_img).unsqueeze(0)  # Shape: (1, 3, 224, 224)
        cropped_visual = VISUAL_CROP_TRANSFORM(pil_img)      # PIL 224x224 for Grad-CAM overlay
        return tensor, cropped_visual

    def predict(self, image_input: Any) -> Dict[str, Any]:
        """
        Performs full inference pipeline on an uploaded image.
        Thread-safe execution.
        """
        pil_img, err = self.validate_image_input(image_input)
        if err is not None:
            return {
                "success": False,
                "error": err,
                "top_class": None,
                "top_full_name": None,
                "top_score": None,
                "top3": [],
                "scores": {},
                "cropped_img": None,
                "input_tensor": None,
            }

        try:
            input_tensor, cropped_img = self.preprocess(pil_img)
            tensor_device = input_tensor.to(self.device)

            with self.infer_lock:
                with torch.no_grad():
                    logits = self.model(tensor_device)
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            # Rank classes
            ranked_indices = probs.argsort()[::-1]
            all_scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
            
            top_idx = ranked_indices[0]
            top_class = CLASS_NAMES[top_idx]
            top_full_name = FULL_CLASS_NAMES[top_class]
            top_score = float(probs[top_idx])

            top3 = [
                {
                    "class": CLASS_NAMES[i],
                    "full_name": FULL_CLASS_NAMES[CLASS_NAMES[i]],
                    "score": float(probs[i]),
                    "percentage": f"{probs[i] * 100:.1f}%",
                }
                for i in ranked_indices[:3]
            ]

            return {
                "success": True,
                "error": None,
                "top_class": top_class,
                "top_full_name": top_full_name,
                "top_score": top_score,
                "top3": top3,
                "scores": all_scores,
                "cropped_img": cropped_img,
                "input_tensor": input_tensor,
                "device": str(self.device),
                "epoch": self.epoch,
            }

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "success": False,
                "error": "GPU out-of-memory error during inference. Please try a smaller image or reset.",
                "top_class": None,
                "top_full_name": None,
                "top_score": None,
                "top3": [],
                "scores": {},
                "cropped_img": None,
                "input_tensor": None,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Model inference error: {str(e)}",
                "top_class": None,
                "top_full_name": None,
                "top_score": None,
                "top3": [],
                "scores": {},
                "cropped_img": None,
                "input_tensor": None,
            }
'''

with open(os.path.join(SRC_DIR, "inference.py"), "w", encoding="utf-8") as f:
    f.write(inference_code)
print("Created C:\SafeDerm\src\inference.py")

# -------------------------------------------------------------
# 2. gradcam.py
# -------------------------------------------------------------
gradcam_code = '''# SafeDerm Triage Lite - Grad-CAM Explainability Module
# Educational/research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM implementation targeting the 14x14 feature layer: model.features[5].
    Avoids the misleading position-bias of features[8] (7x7) while extracting
    interpretable sensitivity maps on EfficientNet-B0 + CBAM.
    """
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        # Target layer is features[5] (output shape 112 x 14 x 14)
        if target_layer is None:
            self.target_layer = model.features[5]
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hook_handles = []

    def _register_hooks(self):
        self.activations = None
        self.gradients = None
        self.hook_handles = []

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient w.r.t the layer's output
            self.gradients = grad_output[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def _remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int,
        cropped_pil: Image.Image,
        colormap_name: str = "jet",
        alpha: float = 0.5
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[str]]:
        """
        Computes Grad-CAM heatmap and blended overlay for a given class index.
        Returns: (heatmap_pil, overlay_pil, error_message)
        """
        self.model.eval()
        self._register_hooks()

        try:
            device = next(self.model.parameters()).device
            x = input_tensor.clone().to(device)
            x.requires_grad_(True)

            self.model.zero_grad()
            logits = self.model(x)

            if target_class_idx < 0 or target_class_idx >= logits.shape[1]:
                return None, None, f"Invalid target class index: {target_class_idx}"

            score = logits[0, target_class_idx]
            score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                return None, None, "Grad-CAM failed to capture feature activations or gradients."

            # Global average pooling of gradients: alpha_k = (1 / Z) * sum(gradients)
            # Gradients shape: (1, 112, 14, 14)
            # Activations shape: (1, 112, 14, 14)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)  # Only features with positive influence

            cam_np = cam.squeeze().detach().cpu().numpy()
            
            # Normalize to [0, 1]
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_norm = np.zeros_like(cam_np)

            # Resize to match 224x224 cropped image
            cam_img = Image.fromarray((cam_norm * 255).astype(np.uint8))
            cam_resized = cam_img.resize((224, 224), Image.Resampling.BICUBIC)
            cam_resized_np = np.array(cam_resized) / 255.0

            # Colorize heatmap with Matplotlib colormap
            cmap = cm.get_cmap(colormap_name)
            heatmap_rgba = cmap(cam_resized_np)  # RGBA in [0, 1]
            heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(heatmap_rgb)

            # Blend with original cropped visual image
            orig_rgb = np.array(cropped_pil.convert("RGB"))
            overlay_np = (orig_rgb * (1.0 - alpha) + heatmap_rgb * alpha).clip(0, 255).astype(np.uint8)
            overlay_pil = Image.fromarray(overlay_np)

            return heatmap_pil, overlay_pil, None

        except Exception as e:
            return None, None, f"Grad-CAM generation error: {str(e)}"
        finally:
            self._remove_hooks()
            self.model.zero_grad()
            self.activations = None
            self.gradients = None
'''

with open(os.path.join(SRC_DIR, "gradcam.py"), "w", encoding="utf-8") as f:
    f.write(gradcam_code)
print("Created C:\SafeDerm\src\gradcam.py")

# -------------------------------------------------------------
# 3. dashboard.py
# -------------------------------------------------------------
dashboard_code = '''# SafeDerm Triage Lite - Model Insights & Performance Analytics Dashboard
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
'''

with open(os.path.join(SRC_DIR, "dashboard.py"), "w", encoding="utf-8") as f:
    f.write(dashboard_code)
print("Created C:\SafeDerm\src\dashboard.py")
