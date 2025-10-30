
from argparse import ArgumentParser
from collections import defaultdict

import jax
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from focal.utils.data import load_data_scope
from focal.utils.metrics import Metrics
from focal.utils.losses import default_eval_metrics_names, default_eval_chunked
from focal.utils.eval import eval_step

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple Residual block used in several generator variants."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):  # noqa: D401
        residual = x
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.lrelu(x + residual)


class GeneratorUpsamplingBigger(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.fc_cond = nn.Linear(cond_dim, 8, bias=False)
        self.bn_cond = nn.BatchNorm1d(8)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout(0.3)

        self.fc = nn.Linear(latent_dim + 8, 7 * 7 * 32, bias=False)
        self.bn0 = nn.BatchNorm1d(7 * 7 * 32)

        self.conv_up1 = nn.Conv2d(32, 32, 3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv_up2 = nn.Conv2d(32 + 1, 16, 3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.res2 = ResBlock(16)

        self.conv_up3 = nn.Conv2d(16, 8, 3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(8)

        self.conv_up4 = nn.Conv2d(8, 4, 3, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(4)

        self.res_block1 = ResBlock(4)
        self.res_block2 = ResBlock(4)
        self.res_block3 = ResBlock(4)

        
        self.conv_out = nn.Conv2d(4, 1, 3, padding='same')
        self.act_out = nn.ReLU(inplace=True)

    # ------------------------------------------------------------------
    def _upsample(self, x, scale: int):
        return F.interpolate(x, scale_factor=scale, mode="nearest")

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor, cond: torch.Tensor, approx: torch.Tensor):
        c = self.fc_cond(cond)
        c = self.bn_cond(c)
        c = self.drop(self.lrelu(c))
        x = torch.cat([z, c], dim=1)
        x = self.lrelu(self.bn0(self.fc(x)))
        x = x.view(-1, 32, 7, 7)

        #7 -> 21
        x = self._upsample(x, 3)
        x = self.lrelu(self.bn1(self.conv_up1(x)))

        # 21 -> 105
        x = self._upsample(x, 5)
        x = torch.cat([x, approx], dim=1)
        x = self.lrelu(self.bn2(self.conv_up2(x)))

        x = self.lrelu(self.bn3(self.conv_up3(x)))

        x = self.lrelu(self.bn4(self.conv_up4(x)))

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.act_out(self.conv_out(x))
        return x


class GeneratorUpsamplingBiggerLessChannels(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.fc_cond = nn.Linear(cond_dim, 8, bias=False)
        self.bn_cond = nn.BatchNorm1d(8)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout(0.3)

        self.fc = nn.Linear(latent_dim + 8, 7 * 7 * 16, bias=False)
        self.bn0 = nn.BatchNorm1d(7 * 7 * 16)

        self.conv_up1 = nn.Conv2d(16, 2, 3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(2)

        self.conv_up2 = nn.Conv2d(3, 16, 3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv_up3 = nn.Conv2d(16, 8, 3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(8)

        self.conv_up4 = nn.Conv2d(8, 4, 3, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(4)

        self.conv_out = nn.Conv2d(4, 1, 3, padding='same')
        self.act_out = nn.ReLU(inplace=True)

    # ------------------------------------------------------------------
    def _upsample(self, x, scale: int):
        return F.interpolate(x, scale_factor=scale, mode="nearest")

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor, cond: torch.Tensor, approx: torch.Tensor):
        c = self.fc_cond(cond)
        c = self.bn_cond(c)
        c = self.drop(self.lrelu(c))
        x = torch.cat([z, c], dim=1)
        x = self.lrelu(self.bn0(self.fc(x)))
        x = x.view(-1, 16, 7, 7)

        #7 -> 21
        x = self._upsample(x, 3)
        x = self.lrelu(self.bn1(self.conv_up1(x)))

        # 21 -> 105
        x = self._upsample(x, 5)
        x = torch.cat([x, approx], dim=1) # 3 channels
        x = self.lrelu(self.bn2(self.conv_up2(x)))

        x = self.lrelu(self.bn3(self.conv_up3(x)))

        x = self.lrelu(self.bn4(self.conv_up4(x)))

        x = self.act_out(self.conv_out(x))
        return x



class GeneratorUpsamplingSmallerNoRes(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Conditioning pathway
        self.fc_cond = nn.Linear(cond_dim, 8, bias=False)
        self.bn_cond = nn.BatchNorm1d(8)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Shared dense -> 7×7×256
        self.fc = nn.Linear(latent_dim + 8, 7 * 7 * 64, bias=False)
        self.bn0 = nn.BatchNorm1d(7 * 7 * 64)

        # Convs after upsampling stages
        self.conv_up1 = nn.Conv2d(64, 64, 2, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv_up2 = nn.Conv2d(64 + 1, 32, 2, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv_up3 = nn.Conv2d(32 + 1, 16, 2, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv_up4 = nn.Conv2d(16, 16, 2, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv_mid = nn.Conv2d(16, 8, 3, bias=False)
        self.bn_mid = nn.BatchNorm2d(8)

        self.conv_post = nn.Conv2d(8 + 1, 8, 5, bias=False)
        self.bn_post = nn.BatchNorm2d(8)

        self.conv_out = nn.Conv2d(8, 1, 2)
        self.act_out = nn.LeakyReLU(0.2, inplace=True)

    # ------------------------------------------------------------------
    def _upsample(self, x, scale):
        return F.interpolate(x, scale_factor=scale, mode="nearest")

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor, cond: torch.Tensor, approx: torch.Tensor):
        c = self.lrelu(self.bn_cond(self.fc_cond(cond)))
        x = torch.cat([z, c], dim=1)
        x = self.lrelu(self.bn0(self.fc(x)))
        x = x.view(-1, 64, 7, 7)

        # 7 → 14
        x = self._upsample(x, 2)
        x = self.lrelu(self.bn1(self.conv_up1(x)))

        # 14 → 28
        x = self._upsample(x, 2)
        x = torch.cat([x, self._upsample(approx, 0.27)], dim=1)
        x = self.lrelu(self.bn2(self.conv_up2(x)))

        # 28 → 56
        x = self._upsample(x, 2)
        x = torch.cat([x, self._upsample(approx, 0.54)], dim=1)
        x = self.lrelu(self.bn3(self.conv_up3(x)))

        # 56 → 112
        x = self._upsample(x, 2)
        x = self.lrelu(self.bn4(self.conv_up4(x)))

        # 112 → 110 (3×3 valid)
        x = self.lrelu(self.bn_mid(self.conv_mid(x)))

        # 110 → 106 (5×5 valid)
        x = torch.cat([x, self._upsample(approx, 1.05)], dim=1)
        x = self.lrelu(self.bn_post(self.conv_post(x)))

        # final 2×2 conv → 105×105
        x = self.act_out(self.conv_out(x))
        return x