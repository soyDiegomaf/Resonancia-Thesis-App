import math
import torch
import torch.nn as nn

from config import BASE_CHANNELS, DROPOUT

try:
    from monai.networks.nets import SegResNet
    MONAI_AVAILABLE = True
except Exception:
    MONAI_AVAILABLE = False


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base=16, dropout=0.1):
        super().__init__()

        self.e1 = ConvBlock(in_channels, base, dropout)
        self.e2 = ConvBlock(base, base * 2, dropout)
        self.e3 = ConvBlock(base * 2, base * 4, dropout)
        self.e4 = ConvBlock(base * 4, base * 8, dropout)

        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, 2)
        self.d3 = ConvBlock(base * 8, base * 4, dropout)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, 2)
        self.d2 = ConvBlock(base * 4, base * 2, dropout)

        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, 2)
        self.d1 = ConvBlock(base * 2, base, dropout)

        self.out = nn.Conv3d(base, out_channels, 1)

    def forward(self, x, available=None):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        d3 = self.d3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


class GatedFusionUNet3D(nn.Module):
    def __init__(self, n_modalities=4, out_channels=3, base=16, dropout=0.1):
        super().__init__()

        self.stems = nn.ModuleList(
            [ConvBlock(1, base, dropout) for _ in range(n_modalities)]
        )

        self.gate = nn.Sequential(
            nn.Linear(base, max(4, base // 2)),
            nn.ReLU(),
            nn.Linear(max(4, base // 2), 1),
        )

        self.core = SimpleUNet3D(base, out_channels, base, dropout)

    def forward(self, x, available=None):
        feats = []
        scores = []

        for i, stem in enumerate(self.stems):
            f = stem(x[:, i : i + 1])
            feats.append(f)
            scores.append(self.gate(f.mean(dim=(2, 3, 4))).squeeze(1))

        scores = torch.stack(scores, dim=1)

        if available is not None:
            scores = scores.masked_fill(available <= 0, -1e4)

        w = torch.softmax(scores, dim=1)

        fused = sum(
            feats[i] * w[:, i].view(-1, 1, 1, 1, 1)
            for i in range(len(feats))
        )

        return self.core(fused)


class HeMISFusionUNet3D(nn.Module):
    def __init__(self, n_modalities=4, out_channels=3, base=16, dropout=0.1):
        super().__init__()

        self.stems = nn.ModuleList(
            [ConvBlock(1, base, dropout) for _ in range(n_modalities)]
        )

        self.core = SimpleUNet3D(base * 2, out_channels, base, dropout)

    def forward(self, x, available=None):
        feats = torch.stack(
            [stem(x[:, i : i + 1]) for i, stem in enumerate(self.stems)],
            dim=1,
        )

        if available is None:
            available = torch.ones(x.shape[0], feats.shape[1], device=x.device)

        mask = available.view(x.shape[0], feats.shape[1], 1, 1, 1, 1)

        count = mask.sum(dim=1).clamp_min(1.0)

        mean = (feats * mask).sum(dim=1) / count
        var = (((feats - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / count

        return self.core(torch.cat([mean, var], dim=1))


class AttentionFusionUNet3D(nn.Module):
    def __init__(self, n_modalities=4, out_channels=3, base=16, dropout=0.1):
        super().__init__()

        self.n_modalities = n_modalities
        self.base = base

        self.stems = nn.ModuleList(
            [ConvBlock(1, base, dropout) for _ in range(n_modalities)]
        )

        self.q = nn.Conv3d(base, base, 1)
        self.k = nn.Conv3d(base, base, 1)
        self.v = nn.Conv3d(base, base, 1)

        self.fuse = nn.Conv3d(base, base, 1)

        self.core = SimpleUNet3D(base, out_channels, base, dropout)

    def forward(self, x, available=None):
        feats = [stem(x[:, i : i + 1]) for i, stem in enumerate(self.stems)]

        q = torch.stack([self.q(f) for f in feats], dim=1)
        k = torch.stack([self.k(f) for f in feats], dim=1)
        v = torch.stack([self.v(f) for f in feats], dim=1)

        _, _, c, _, _, _ = q.shape

        qv = q.permute(0, 3, 4, 5, 1, 2)
        kv = k.permute(0, 3, 4, 5, 1, 2)
        vv = v.permute(0, 3, 4, 5, 1, 2)

        scores = torch.matmul(qv, kv.transpose(-1, -2)) / math.sqrt(c)

        if available is not None:
            mask = available[:, None, None, None, None, :]
            scores = scores.masked_fill(mask <= 0, -1e4)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, vv).mean(dim=-2).permute(0, 4, 1, 2, 3)

        return self.core(self.fuse(out))


def build_model(name: str) -> nn.Module:
    name = name.lower()

    if name == "early_fusion_unet":
        return SimpleUNet3D(4, 3, BASE_CHANNELS, DROPOUT)

    if name == "gated_fusion_unet":
        return GatedFusionUNet3D(4, 3, BASE_CHANNELS, DROPOUT)

    if name == "hemis_fusion_unet":
        return HeMISFusionUNet3D(4, 3, BASE_CHANNELS, DROPOUT)

    if name == "attention_fusion_unet":
        return AttentionFusionUNet3D(4, 3, BASE_CHANNELS, DROPOUT)

    if name == "segresnet":
        if not MONAI_AVAILABLE:
            raise RuntimeError("MONAI is not installed. Run: uv pip install monai")

        return SegResNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            init_filters=BASE_CHANNELS,
            dropout_prob=DROPOUT,
        )

    raise ValueError(f"Unknown model architecture: {name}")


def load_checkpoint(model: nn.Module, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model", checkpoint)

    model.load_state_dict(state_dict, strict=True)

    return model