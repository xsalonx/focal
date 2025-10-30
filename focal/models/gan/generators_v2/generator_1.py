import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------ helpers ------------------------

class CBN2d(nn.Module):
    """
    Conditional BatchNorm2d: BN without affine; gamma/beta predicted from cond.
    Initialized as identity: y = BN(x); out = (1+Δγ(cond)) * y + Δβ(cond)
    """
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta  = nn.Linear(cond_dim, num_features)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        y = self.bn(x)
        g = 1 + self.gamma(cond).unsqueeze(-1).unsqueeze(-2)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-2)
        return g * y + b


def conv3x3(in_ch, out_ch, bias=False):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias)

def conv1x1(in_ch, out_ch, bias=False):
    return nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=bias)


class ResBlockUpCBN(nn.Module):
    """
    Pre-activation ResNet block with upsampling in the residual branch.
    Shape: [B, Cin, H, W] -> upsample(scale) -> [B, Cout, scale*H, scale*W]
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, scale: int):
        super().__init__()
        self.scale = scale
        self.cbn1 = CBN2d(in_ch, cond_dim)
        self.cbn2 = CBN2d(out_ch, cond_dim)
        self.conv1 = conv3x3(in_ch, out_ch, bias=False)
        self.conv2 = conv3x3(out_ch, out_ch, bias=False)
        self.skip  = conv1x1(in_ch, out_ch, bias=False) if in_ch != out_ch else nn.Identity()
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, cond):
        h = self.act(self.cbn1(x, cond))
        h = F.interpolate(h, scale_factor=self.scale, mode="nearest")
        h = self.conv1(h)
        h = self.act(self.cbn2(h, cond))
        h = self.conv2(h)

        s = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        s = self.skip(s)
        return h + s


class ResBlockCBN(nn.Module):
    """Pre-activation ResNet block without spatial rescaling."""
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.cbn1 = CBN2d(ch, cond_dim)
        self.cbn2 = CBN2d(ch, cond_dim)
        self.conv1 = conv3x3(ch, ch, bias=False)
        self.conv2 = conv3x3(ch, ch, bias=False)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, cond):
        h = self.act(self.cbn1(x, cond))
        h = self.conv1(h)
        h = self.act(self.cbn2(h, cond))
        h = self.conv2(h)
        return x + h

# ------------------------ generator ------------------------

class GeneratorUpsamplingBigger_v2_1(nn.Module):
    """
    7x7 -> (x3) 21x21 -> (x5) 105x105; condition used in every block via CBN/FiLM.
    Output uses Softplus for non-negative, non-dead gradients.
    """
    def __init__(self, latent_dim: int, cond_dim: int, cond_embed: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.cond_embed = cond_embed

        # condition embedding reused by all CBNs
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_embed),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cond_embed, cond_embed),
        )

        # fuse z and embedded cond at input
        self.fc = nn.Linear(latent_dim + cond_embed, 7 * 7 * 256, bias=False)
        self.bn0 = CBN2d(256, cond_embed)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # ResNet upsample blocks
        self.up1 = ResBlockUpCBN(256, 256, cond_embed, scale=3)  # 7 -> 21
        self.up2 = ResBlockUpCBN(256, 128, cond_embed, scale=5)  # 21 -> 105

        # channel reductions + refinement at full resolution
        self.to64   = conv3x3(128, 64, bias=False)
        self.cbn64  = CBN2d(64, cond_embed)
        self.ref64a = ResBlockCBN(64, cond_embed)

        self.to32   = conv3x3(64, 32, bias=False)
        self.cbn32  = CBN2d(32, cond_embed)
        self.ref32a = ResBlockCBN(32, cond_embed)
        self.ref32b = ResBlockCBN(32, cond_embed)
        self.ref32c = ResBlockCBN(32, cond_embed)

        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)
        self.softplus = nn.Softplus()

        nn.init.zeros_(self.conv_out.bias)

    def forward(self, z: torch.Tensor, cond: torch.Tensor):
        c = self.cond_mlp(cond)                               # [B, E]
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, 256, 7, 7)
        x = self.act(self.bn0(x, c))

        x = self.up1(x, c)                                    # [B,256,21,21]
        x = self.up2(x, c)                                    # [B,128,105,105]

        x = self.act(self.cbn64(self.to64(x), c))             # [B,64,105,105]
        x = self.ref64a(x, c)

        x = self.act(self.cbn32(self.to32(x), c))             # [B,32,105,105]
        x = self.ref32a(x, c)
        x = self.ref32b(x, c)
        x = self.ref32c(x, c)

        x = self.conv_out(x)
        return self.softplus(x)
