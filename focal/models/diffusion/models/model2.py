import torch
from diffusers import UNet2DConditionModel, get_cosine_schedule_with_warmup

import os
script_path = os.path.abspath(__file__)

code_snippet = ""
with open(script_path) as f:
    code_snippet = f.read()

from focal.models import RESPONSE_SHAPE, PARTICLE_SHAPE


model = UNet2DConditionModel(
    sample_size=RESPONSE_SHAPE[0],
    in_channels=RESPONSE_SHAPE[-1],
    out_channels=RESPONSE_SHAPE[-1],
    layers_per_block=2,
    block_out_channels=(16, 32, 64, 128),
    down_block_types=(
        'DownBlock2D',
        'DownBlock2D',
        'DownBlock2D',
        'DownBlock2D',
    ),
    mid_block_type='UNetMidBlock2D',
    up_block_types=(
        'UpBlock2D',
        'UpBlock2D',
        'UpBlock2D',
        'UpBlock2D',
    ),
    encoder_hid_dim=PARTICLE_SHAPE[0],
    norm_num_groups=8
)
