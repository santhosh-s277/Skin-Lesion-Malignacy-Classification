# SafeDerm Triage Lite - Grad-CAM Explainability Module
# Educational/research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Optional, Dict, Any


class GradCAM:
    """
    Mathematically rigorous Grad-CAM implementation targeting model.features[5] (14x14 receptive field).
    Calculates class-discriminative spatial sensitivity maps using:
        alpha_k^c = (1 / HW) * sum_{i,j} (d y^c / d A_{i,j}^k)
        L_GradCAM^c = ReLU(sum_k alpha_k^c * A^k)
    Uses continuous float bilinear tensor interpolation to match input resolution (224x224),
    preventing 8-bit quantization noise and bicubic edge ringing.
    Guarantees hook cleanup and zero memory leakage.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        # Target layer is features[5] (Stage 5 InvertedResidual + CBAM, output shape: [B, 112, 14, 14])
        if target_layer is None:
            self.target_layer = model.features[5]
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hook_handles = []

    def _register_hooks(self):
        self._remove_hooks()
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient of the target class logit w.r.t layer output
            self.gradients = grad_output[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def _remove_hooks(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.hook_handles.clear()

    def compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        Computes raw Grad-CAM tensor and layer statistics.
        Returns:
            cam_224: torch.Tensor shape [224, 224] in [0, 1] float
            logits: torch.Tensor shape [1, 8]
            target_class_idx: int (the class explained)
            alpha_weights: torch.Tensor shape [112] (channel importance weights)
        """
        self.model.eval()
        self._register_hooks()

        try:
            device = next(self.model.parameters()).device
            x = input_tensor.clone().to(device)

            self.model.zero_grad()
            logits = self.model(x)

            if target_class_idx is None:
                target_class_idx = logits.argmax(dim=1).item()

            if target_class_idx < 0 or target_class_idx >= logits.shape[1]:
                raise ValueError(f"Invalid target_class_idx {target_class_idx}. Logits shape: {logits.shape}")

            # Target scalar is the uncalibrated raw class logit y^c
            target_scalar = logits[0, target_class_idx]
            target_scalar.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM failed to capture feature activations or gradients.")

            # Squeeze batch dimension: [C, H, W]
            act = self.activations[0]
            grad = self.gradients[0]

            # Spatially averaged gradient weights: alpha_k = (1 / HW) * sum(grad_k)
            # Shape: [C, 1, 1]
            weights = torch.mean(grad, dim=(1, 2), keepdim=True)

            # Class-specific map: L^c = ReLU(sum_k alpha_k * A^k)
            # Shape: [1, 1, H, W]
            cam = torch.sum(weights * act, dim=0, keepdim=True).unsqueeze(0)
            cam = F.relu(cam)

            # High-precision Bilinear interpolation to 224x224 directly in float tensor
            cam_224 = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam_224 = cam_224.squeeze()  # Shape: [224, 224]

            # Safe normalization to [0, 1]
            c_min = cam_224.min()
            c_max = cam_224.max()
            if (c_max - c_min) > 1e-8:
                cam_norm = (cam_224 - c_min) / (c_max - c_min)
            else:
                cam_norm = torch.zeros_like(cam_224)

            return cam_norm.detach().cpu(), logits.detach().cpu(), target_class_idx, weights.squeeze().detach().cpu()

        finally:
            self._remove_hooks()
            self.model.zero_grad()
            self.activations = None
            self.gradients = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int],
        cropped_pil: Image.Image,
        colormap_name: str = "jet",
        alpha: float = 0.5
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[str]]:
        """
        Computes Grad-CAM heatmap and blended overlay for a given class index.
        Returns: (heatmap_pil, overlay_pil, error_message)
        """
        try:
            cam_tensor, logits, actual_idx, weights = self.compute_cam(input_tensor, target_class_idx)
            cam_np = cam_tensor.numpy()

            # Colorize with Matplotlib continuous colormap
            cmap = matplotlib.colormaps[colormap_name]
            heatmap_rgba = cmap(cam_np)  # Float array in [0, 1]
            heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(heatmap_rgb)

            # Blend with un-normalized visual crop (guaranteed 224x224 RGB)
            orig_rgb = np.array(cropped_pil.convert("RGB"))
            if orig_rgb.shape[:2] != (224, 224):
                cropped_pil = cropped_pil.resize((224, 224), Image.Resampling.BICUBIC)
                orig_rgb = np.array(cropped_pil.convert("RGB"))

            overlay_np = (orig_rgb * (1.0 - alpha) + heatmap_rgb * alpha).clip(0, 255).astype(np.uint8)
            overlay_pil = Image.fromarray(overlay_np)

            return heatmap_pil, overlay_pil, None

        except Exception as e:
            return None, None, f"Grad-CAM calculation notice: {str(e)}"
