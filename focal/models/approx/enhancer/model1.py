import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3x3-BN-ReLU) x2 with same padding."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    """Upscaling with bilinear resize (odd-safe), 1x1 reduce, concat skip, then double conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # Resize decoder feature to match skip resolution precisely
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UnetEnhancer(nn.Module):
    """Minimal U-Net-like model for (N,1,105,105) -> (N,1,105,105)."""
    def __init__(self, in_channels=1, out_channels=1, base_ch=4):
        super().__init__()
        b = base_ch
        self.inc   = DoubleConv(in_channels, b)     # HxW -> HxW
        self.down1 = Down(b, b*2)                   # H/2 x W/2
        self.down2 = Down(b*2, b*4)                 # H/4 x W/4

        self.up1   = Up(b*4, b*2, b*2)              # back to H/2 x W/2
        self.up2   = Up(b*2, b, b)                  # back to H x W
        self.outc  = nn.Conv2d(b, out_channels, 1)  # final 1x1

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x  = self.up1(x3, x2)
        x  = self.up2(x,  x1)
        return self.outc(x)
