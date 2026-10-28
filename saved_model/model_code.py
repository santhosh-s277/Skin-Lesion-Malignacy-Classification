import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# stage index -> output channels of that EfficientNet-B0 stage
CBAM_STAGES = {3: 40, 5: 112, 7: 320}
REDUCTION = 16
MIN_HIDDEN = 8


class ChannelAttention(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, x):
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        mx = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.channel = ChannelAttention(channels, hidden)
        self.spatial = SpatialAttention(7)

    def forward(self, x):
        return self.spatial(self.channel(x))


def build_model(num_classes=8):
    # Random init, no pretrained download. Load weights separately.
    model = efficientnet_b0(weights=None, num_classes=num_classes)
    for idx, ch in CBAM_STAGES.items():
        hidden = max(MIN_HIDDEN, ch // REDUCTION)
        model.features[idx] = nn.Sequential(model.features[idx], CBAM(ch, hidden))
    return model
