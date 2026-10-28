# Helper to update gradcam.py with Matplotlib 3.11+ colormap API
import os

gradcam_path = r"C:\SafeDerm\src\gradcam.py"

gradcam_code = '''# SafeDerm Triage Lite - Grad-CAM Explainability Module
# Educational/research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

            # Colorize heatmap with Matplotlib 3.8+ colormap API
            cmap = matplotlib.colormaps[colormap_name]
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

with open(gradcam_path, "w", encoding="utf-8") as f:
    f.write(gradcam_code)
print(f"Updated {gradcam_path}")
