# diagnose_gradcam_details.py
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, r"C:\SafeDerm\src")
sys.path.insert(0, r"C:\SafeDerm\saved_model")

from inference import InferenceService, CLASS_NAMES

def diagnose():
    service = InferenceService.get_instance()
    model = service.model
    device = service.device

    img = Image.open(r"C:\SafeDerm\sample_test_image.jpg")
    tensor, visual = service.preprocess(img)
    tensor = tensor.to(device)

    print("=== MODEL LOGITS & PREDICTION ===")
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    for i, c in enumerate(CLASS_NAMES):
        print(f"[{i}] {c:5s}: logit = {logits[0, i].item():8.4f}, prob = {probs[i].item()*100:6.2f}%")

    print("\n=== LAYER INVESTIGATION ===")
    target_layers = {
        "features[4] (14x14)": model.features[4],
        "features[5][0] (MBConv, 14x14)": model.features[5][0],
        "features[5][1] (CBAM, 14x14)": model.features[5][1],
        "features[5] (Full Stage 5, 14x14)": model.features[5],
        "features[6] (7x7)": model.features[6],
        "features[7] (Full Stage 7, 7x7)": model.features[7],
        "features[8] (Final Conv, 7x7)": model.features[8],
    }

    for name, layer in target_layers.items():
        acts = []
        grads = []

        def f_hook(m, inp, out):
            acts.append(out)

        def b_hook(m, gin, gout):
            grads.append(gout[0])

        h1 = layer.register_forward_hook(f_hook)
        h2 = layer.register_full_backward_hook(b_hook)

        x = tensor.clone().requires_grad_(True)
        model.zero_grad()
        out_logits = model(x)
        score = out_logits[0, 1]  # NV logit
        score.backward()

        h1.remove()
        h2.remove()
        model.zero_grad()

        A = acts[0]
        G = grads[0]
        weights = torch.mean(G, dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(weights * A, dim=1, keepdim=True))

        cam_np = cam[0, 0].detach().cpu().numpy()
        print(f"\nLayer: {name}")
        print(f"  Activation shape: {list(A.shape)}, mean: {A.mean().item():.4f}, min: {A.min().item():.4f}, max: {A.max().item():.4f}")
        print(f"  Gradient shape:   {list(G.shape)}, mean: {G.mean().item():.6f}, min: {G.min().item():.6f}, max: {G.max().item():.6f}")
        print(f"  CAM shape:        {list(cam_np.shape)}, min: {cam_np.min():.4f}, max: {cam_np.max():.4f}")
        # Spatial center vs border ratio
        h, w = cam_np.shape
        center_mask = np.zeros((h, w), dtype=bool)
        center_mask[h//4: 3*h//4, w//4: 3*w//4] = True
        center_mean = cam_np[center_mask].mean()
        border_mean = cam_np[~center_mask].mean()
        print(f"  Center mean: {center_mean:.4f}, Border mean: {border_mean:.4f}, Center/Border ratio: {center_mean/(border_mean+1e-8):.2f}")

if __name__ == "__main__":
    diagnose()
