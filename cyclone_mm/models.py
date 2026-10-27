"""Tabular baseline and image + tabular fusion model."""
import torch
import torch.nn as nn
from torchvision import models as tvm


class TabularEncoder(nn.Module):
    """MLP: Linear -> BatchNorm -> ReLU -> Dropout, repeated."""

    def __init__(self, in_dim: int, hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class TabularModel(nn.Module):
    """Baseline: tabular features only."""

    def __init__(self, in_dim: int, hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        self.encoder = TabularEncoder(in_dim, hidden, dropout)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, tabular, image=None):
        return self.head(self.encoder(tabular))


BACKBONES = {
    # name: (constructor, ImageNet weights, attribute holding the classifier)
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
    "efficientnet_b2": (tvm.efficientnet_b2, tvm.EfficientNet_B2_Weights.IMAGENET1K_V1, "classifier"),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, "fc"),
}


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    ctor, weights, head_attr = BACKBONES[name]
    m = ctor(weights=weights if pretrained else None)
    head = getattr(m, head_attr)
    out_dim = head[-1].in_features if isinstance(head, nn.Sequential) else head.in_features
    setattr(m, head_attr, nn.Identity())  # m(x) now returns the pooled embedding
    return m, out_dim


class MultimodalModel(nn.Module):
    """CNN image embedding || MLP tabular embedding -> fusion head -> wind (kt)."""

    def __init__(self, tab_dim: int, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 img_proj: int = 256, tab_hidden=(128, 64), fusion_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone, pretrained)
        self.img_proj = nn.Sequential(nn.Linear(feat_dim, img_proj), nn.BatchNorm1d(img_proj),
                                      nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.tab_encoder = TabularEncoder(tab_dim, tab_hidden, dropout=0.2)
        self.fusion = nn.Sequential(
            nn.Linear(img_proj + self.tab_encoder.out_dim, fusion_hidden), nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(fusion_hidden, 1))

    def forward(self, tabular, image):
        z = torch.cat([self.img_proj(self.backbone(image)), self.tab_encoder(tabular)], dim=1)
        return self.fusion(z)

    def param_groups(self, lr_backbone: float, lr_head: float):
        head = [p for n, p in self.named_parameters() if not n.startswith("backbone.")]
        return [{"params": self.backbone.parameters(), "lr": lr_backbone},
                {"params": head, "lr": lr_head}]
