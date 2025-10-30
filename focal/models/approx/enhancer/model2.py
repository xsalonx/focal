import torch
import torch.nn as nn

# ===== blocks with BatchNorm =====

class ConvBlockBN(nn.Module):
    # (Conv3x3 -> BN -> SiLU) x2, keeps HxW
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class DownConvBN(nn.Module):
    # downsample with stride-2 conv, then BN+SiLU
    # sizes: 105 -> 53 -> 27 -> 14 (k=3, s=2, p=1)
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class UpConvBN(nn.Module):
    # upsample with ConvTranspose2d, then BN+SiLU
    # exact inverse of DownConvBN: (H-1)*2 - 2*1 + 3 + 0
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=0, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.act   = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        return self.act(x)

# ===== the model =====

class UNetEnhancer(nn.Module):
    """
    U-Net-ish with only convolutions, transposed-convs, activations, and BatchNorm.
    Input:  (N, 1, 105, 105) -> Output: (N, 1, 105, 105)
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1  = ConvBlockBN(in_ch, base_ch)           # 105x105
        self.down1 = DownConvBN(base_ch, base_ch*2)           # 53x53

        self.enc2  = ConvBlockBN(base_ch*2, base_ch*2)        # 53x53
        self.down2 = DownConvBN(base_ch*2, base_ch*4)         # 27x27

        self.enc3  = ConvBlockBN(base_ch*4, base_ch*4)        # 27x27
        self.down3 = DownConvBN(base_ch*4, base_ch*8)         # 14x14

        # Bottleneck
        self.bottleneck = ConvBlockBN(base_ch*8, base_ch*8)   # 14x14

        # Decoder
        self.up3  = UpConvBN(base_ch*8, base_ch*4)            # 27x27
        self.dec3 = ConvBlockBN(base_ch*4 + base_ch*4, base_ch*4)

        self.up2  = UpConvBN(base_ch*4, base_ch*2)            # 53x53
        self.dec2 = ConvBlockBN(base_ch*2 + base_ch*2, base_ch*2)

        self.up1  = UpConvBN(base_ch*2, base_ch)              # 105x105
        self.dec1 = ConvBlockBN(base_ch + base_ch, base_ch)

        self.head = nn.Conv2d(base_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # 105
        x  = self.down1(e1)      # 53

        e2 = self.enc2(x)        # 53
        x  = self.down2(e2)      # 27

        e3 = self.enc3(x)        # 27
        x  = self.down3(e3)      # 14

        # Bottleneck
        x  = self.bottleneck(x)  # 14

        # Decoder
        x  = self.up3(x)         # 27
        x  = torch.cat([x, e3], dim=1)
        x  = self.dec3(x)

        x  = self.up2(x)         # 53
        x  = torch.cat([x, e2], dim=1)
        x  = self.dec2(x)

        x  = self.up1(x)         # 105
        x  = torch.cat([x, e1], dim=1)
        x  = self.dec1(x)

        return self.head(x)
