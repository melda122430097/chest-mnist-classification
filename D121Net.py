# EffNet.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetSmall(nn.Module):
    """
    ResNet-18 yang disesuaikan untuk ChestMNIST (28x28, grayscale),
    dan kompatibel dengan train.py kamu:
      - num_classes == 2 -> output (N,1) untuk BCEWithLogitsLoss
      - num_classes > 2  -> output (N,num_classes) untuk CrossEntropyLoss
    """
    def __init__(self, in_channels=1, num_classes=2, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Ganti conv pertama agar mendukung grayscale 1-channel
        if in_channels != 3:
            old_conv = self.backbone.conv1
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=False)
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            self.backbone.conv1 = new_conv

        # Hilangkan maxpool agar fitur tidak mengecil menjadi 0
        self.backbone.maxpool = nn.Identity()

        # Replace classifier
        in_features = self.backbone.fc.in_features
        if num_classes == 2:
            self.backbone.fc = nn.Linear(in_features, 1)
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Quick test
if __name__ == "__main__":
    model = ResNetSmall(in_channels=1, num_classes=2, pretrained=True)
    x = torch.randn(4,1,28,28)
    y = model(x)
    print("Output shape:", y.shape)  # Expected: (4,1)
    print("Model berjalan ✅")
