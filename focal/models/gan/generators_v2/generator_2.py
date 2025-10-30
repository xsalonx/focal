import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- FiLM na GroupNorm (działa lepiej przy małych batchach) ----------
class FiLM2d(nn.Module):
    def __init__(self, ch: int, cond_dim: int, groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch, affine=False)
        self.gamma = nn.Linear(cond_dim, ch)
        self.beta  = nn.Linear(cond_dim, ch)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        y = self.norm(x)
        g = 1 + self.gamma(cond).unsqueeze(-1).unsqueeze(-2)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-2)
        return y * g + b

def conv3x3(cin, cout, bias=False):
    return nn.Conv2d(cin, cout, 3, padding=1, bias=bias)

def conv1x1(cin, cout, bias=False):
    return nn.Conv2d(cin, cout, 1, bias=bias)

# ---------- ResNet up z PixelShuffle (sub-pixel) ----------
class ResBlockUpShuffle(nn.Module):
    """
    Pre-activation: FiLM -> act -> conv -> PixelShuffle(r) -> FiLM -> act -> conv
    Skip: 1x1 conv -> PixelShuffle(r)
    Wejście [B,Cin,H,W], wyjście [B,Cout,rH,rW]
    """
    def __init__(self, cin: int, cout: int, cond_dim: int, r: int, groups: int = 32):
        super().__init__()
        self.r = r
        self.film1 = FiLM2d(cin, cond_dim, groups)
        self.film2 = FiLM2d(cout, cond_dim, groups)
        # najpierw podnosimy kanały x r^2, potem PixelShuffle ↓ kanały
        self.conv1 = conv3x3(cin, cout * r * r, bias=False)
        self.ps1   = nn.PixelShuffle(r)
        self.conv2 = conv3x3(cout, cout, bias=False)
        # skip
        self.skip  = conv1x1(cin, cout * r * r, bias=False)
        self.pss   = nn.PixelShuffle(r)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, c):
        h = self.act(self.film1(x, c))
        h = self.ps1(self.conv1(h))
        h = self.act(self.film2(h, c))
        h = self.conv2(h)

        s = self.pss(self.skip(x))
        return h + s

class ResBlockFiLM(nn.Module):
    """ Zwykły ResNet blok z FiLM + GroupNorm, bez zmiany rozmiaru. """
    def __init__(self, ch: int, cond_dim: int, groups: int = 32):
        super().__init__()
        self.f1 = FiLM2d(ch, cond_dim, groups)
        self.f2 = FiLM2d(ch, cond_dim, groups)
        self.c1 = conv3x3(ch, ch, bias=False)
        self.c2 = conv3x3(ch, ch, bias=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, c):
        h = self.act(self.f1(x, c))
        h = self.c1(h)
        h = self.act(self.f2(h, c))
        h = self.c2(h)
        return x + h

# ---------- Generator: 105x105, 1 kanał, zero atencji ----------
class GeneratorUpsamplingBigger_v2_2(nn.Module):
    """
    7x7 -> (×3) 21x21 -> (×5) 105x105
    Kondycja w każdym bloku (FiLM), upsampling przez PixelShuffle,
    wyjście Softplus (>=0, miękkie gradienty).
    """
    def __init__(self, latent_dim: int, cond_dim: int,
                 base_ch: int = 256, cond_embed: int = 64, groups: int = 32):
        super().__init__()
        self.base_ch = base_ch
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.softplus = nn.Softplus()

        # embed kondycji, używany przez FiLM wszędzie
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_embed),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cond_embed, cond_embed),
        )

        # wejście: z + c_emb -> 7x7xC
        self.fc = nn.Linear(latent_dim + cond_embed, base_ch * 7 * 7, bias=False)
        self.f0 = FiLM2d(base_ch, cond_embed, groups)

        # 7 -> 21 (r=3), 256 -> 256
        self.up1 = ResBlockUpShuffle(base_ch, base_ch, cond_embed, r=3, groups=groups)
        # 21 -> 105 (r=5), 256 -> 128
        self.up2 = ResBlockUpShuffle(base_ch, base_ch // 2, cond_embed, r=5, groups=groups)

        # rafinacja na pełnej rozdzielczości
        self.to64  = conv3x3(base_ch // 2, 64, bias=False)
        self.f64   = FiLM2d(64, cond_embed, groups)
        self.ref64 = ResBlockFiLM(64, cond_embed, groups)

        self.to32  = conv3x3(64, 32, bias=False)
        self.f32   = FiLM2d(32, cond_embed, groups)
        self.ref32a = ResBlockFiLM(32, cond_embed, groups)
        self.ref32b = ResBlockFiLM(32, cond_embed, groups)

        self.out = nn.Conv2d(32, 1, 3, padding=1)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, cond: torch.Tensor):
        c = self.cond_mlp(cond)                        # [B, E]
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, self.base_ch, 7, 7)             # [B, C=256, 7, 7]
        x = self.act(self.f0(x, c))

        x = self.up1(x, c)                            # [B, 256, 21, 21]
        x = self.up2(x, c)                            # [B, 128, 105, 105]

        x = self.act(self.f64(self.to64(x), c))       # [B, 64, 105, 105]
        x = self.ref64(x, c)

        x = self.act(self.f32(self.to32(x), c))       # [B, 32, 105, 105]
        x = self.ref32a(x, c)
        x = self.ref32b(x, c)

        x = self.out(x)
        return self.softplus(x)
