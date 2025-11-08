# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNetClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        # Load DenseNet-121
        self.backbone = models.densenet121(pretrained=False)

        # Ubah input conv kalau channel != 3
        if in_channels != 3:
            old_conv = self.backbone.features.conv0
            self.backbone.features.conv0 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        # Ubah classifier jumlah output
        in_features = self.backbone.classifier.in_features
        out_features = 1 if num_classes == 2 else num_classes
        self.backbone.classifier = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Resize agar input 28x28 tetap bisa masuk DenseNet
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)


__all__ = ["DenseNetClassifier"]


# --- Bagian pengujian ---
if __name__ == '__main__':
    model = DenseNetClassifier(in_channels=1, num_classes=2)
    dummy = torch.randn(8, 1, 28, 28)
    print(model)
    out = model(dummy)
    print("Output shape:", out.shape)
    print("Model berjalan ✅")
