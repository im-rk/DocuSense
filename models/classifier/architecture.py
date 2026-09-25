"""Model 1: page-type classifier. Small ResNet trained from scratch (no pretrained weights)."""
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Basic residual block: conv-BN-ReLU-conv-BN + skip connection."""

    def __init__(self, i, o, stride):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o)
        self.c2, self.b2 = nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o)
        self.sc = nn.Identity() if (stride == 1 and i == o) else nn.Sequential(
            nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(y)) + self.sc(x))


def _init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, Block):
            nn.init.zeros_(m.b2.weight)          # each residual block starts as identity: stable from-scratch training


def _head(n_cls):
    return nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(4096, 128), nn.ReLU(inplace=True),
                         nn.Dropout(0.2), nn.Linear(128, n_cls))


class DocClassifierResNet(nn.Module):
    """Input (B,1,256,256) grayscale -> (B,n_cls) logits. ~3.5M parameters."""

    def __init__(self, n_cls=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.stages = nn.Sequential(
            Block(32, 32, 1), Block(32, 32, 1),          # 128x128
            Block(32, 64, 2), Block(64, 64, 1),          # 64x64
            Block(64, 128, 2), Block(128, 128, 1),       # 32x32
            Block(128, 256, 2), Block(256, 256, 1))      # 16x16
        self.pool = nn.AdaptiveAvgPool2d(4)              # 256x4x4: keeps coarse page layout
        self.head = _head(n_cls)
        _init(self)

    def forward(self, x):
        return self.head(self.pool(self.stages(self.stem(x))))


class PlainCNN(nn.Module):
    """Ablation baseline: 5 plain conv-BN-ReLU-pool blocks, no skip connections."""

    def __init__(self, n_cls=5):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256), blk(256, 256),
                                      nn.AdaptiveAvgPool2d(4))
        self.head = _head(n_cls)
        _init(self)

    def forward(self, x):
        return self.head(self.features(x))


def build_classifier(arch="resnet", n_cls=5):
    return {"resnet": DocClassifierResNet, "plain": PlainCNN}[arch](n_cls)
