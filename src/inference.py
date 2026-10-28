# SafeDerm Triage Lite - Model Inference & Preprocessing Service
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
