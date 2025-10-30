
import torch
import torch.nn.functional as F


from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """Mirror of original `prepare_discriminator` (the larger one)."""

    def __init__(self, in_shape: Tuple[int, int, int], cond_dim: int):
        super().__init__()
        channels_no = in_shape[0]
        assert channels_no == 1, "Only single‑channel images supported (port assumption)."

        self.conv1 = nn.Conv2d(channels_no, 16, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.drop = nn.Dropout(0.3)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Dense heads
        # self.fc_img = nn.Linear(64 * (in_shape[0] // 8) * (in_shape[1] // 8), 64)
        self.fc_img = nn.Linear(64 * 14 * 14, 64)
        self.bn_img = nn.BatchNorm1d(64)

        self.fc_cond = nn.Linear(cond_dim, 16)
        self.bn_cond = nn.BatchNorm1d(16)

        self.fc1 = nn.Linear(64 + 16, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc_out = nn.Linear(32, 1)

    # ------------------------------------------------------------------
    def forward(self, img: torch.Tensor, cond: torch.Tensor):  # noqa: D401
        x = self.drop(self.lrelu(self.bn1(self.conv1(img))))
        x = self.drop(self.lrelu(self.bn2(self.conv2(x))))
        x = self.drop(self.lrelu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.drop(self.lrelu(self.bn_img(self.fc_img(x))))

        c = self.drop(self.lrelu(self.bn_cond(self.fc_cond(cond))))
        x = torch.cat([x, c], dim=1)
        x = self.drop(self.lrelu(self.bn4(self.fc1(x))))
        x = self.drop(self.lrelu(self.bn5(self.fc2(x))))
        return self.fc_out(x)  # logits
