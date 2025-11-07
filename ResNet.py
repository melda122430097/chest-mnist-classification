import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class DashNet(nn.Module):
    """
    DashNet: light residual network with global avg pool.
    - Returns (N,1) when num_classes == 2 (compatible with BCEWithLogitsLoss + labels shaped (N,1))
    - Returns (N,C) when num_classes > 2 (compatible with CrossEntropyLoss + labels shaped (N,))
    """
    def __init__(self, in_channels=1, num_classes=2, block=BasicBlock, layers=(2,2,2), base_planes=32):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, base_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_planes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, base_planes, base_planes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_planes, base_planes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_planes*2, base_planes*4, layers[2], stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        out_features = base_planes * 4
        self.fc = nn.Linear(out_features, 1 if num_classes == 2 else num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(in_planes, planes, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)        # (N, C, 1, 1)
        x = torch.flatten(x, 1)       # (N, C)
        x = self.fc(x)                # (N,1) or (N,C)
        return x

# alias / convenience
ResNet = DashNet

if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    model = DashNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print(model)
    dummy = torch.randn(4, IN_CHANNELS, 28, 28)
    print("output shape:", model(dummy).shape)