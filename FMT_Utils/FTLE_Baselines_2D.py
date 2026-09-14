"""Frozen PyFlowVis neural baselines, copied without class-body edits.
Source: PyFlowVis 3040ace90b34483fd80387166f17d97305d7ae45,
FMT_Utils/model_zoo.py; local inherited file used for extraction.
FTLE adaptations use in_ch=1 and documented configuration.
"""
import torch
from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if needed to handle odd sizes
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class ESPCN_SR(nn.Module):
    """Canonical efficient sub-pixel CNN (Shi et al. 2016) for 2-channel flow-map SR,
    matching Jakob et al. 2020: two conv layers (f1,f2; n1,n2 features, ReLU) followed by
    a sub-pixel convolution that upsamples by `upscale`. Input/output are 2-channel images
    (the particle end-position flow map). Fully convolutional -> any H x W at inference."""
    def __init__(self, cfg, upscale: int = 2, in_ch: int = 2):
        super().__init__()
        f1 = int(getattr(cfg.model, 'f1', 3))
        f2 = int(getattr(cfg.model, 'f2', 3))
        n1 = int(getattr(cfg.model, 'n1', 128))
        n2 = int(getattr(cfg.model, 'n2', 128))
        self.upscale = int(upscale)
        self.in_ch = int(in_ch)
        self.conv1 = nn.Conv2d(in_ch, n1, kernel_size=f1, padding=f1 // 2)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=f2 // 2)
        self.conv3 = nn.Conv2d(n2, in_ch * self.upscale * self.upscale, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(self.upscale)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, H, W] -> [B, 2, H*UP, W*UP]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return self.shuffle(x)

class UNet_SR(nn.Module):
    """UNet body (2 downsamples) + sub-pixel upsampling head for 2-channel flow-map SR.
    Same I/O contract as ESPCN_SR (2-ch image -> 2-ch image at k x). Fully convolutional."""
    def __init__(self, cfg, upscale: int = 2, in_ch: int = 2, base: int = 64):
        super().__init__()
        base = int(getattr(cfg.model, 'base', base))
        self.upscale = int(upscale)
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.up1 = Up(base * 4, base * 2)
        self.up2 = Up(base * 2, base)
        self.head = nn.Sequential(
            nn.Conv2d(base, in_ch * self.upscale * self.upscale, kernel_size=3, padding=1),
            nn.PixelShuffle(self.upscale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.head(x)
