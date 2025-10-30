from diffusers import UNet2DConditionModel

from focal.models import RESPONSE_SHAPE, PARTICLE_SHAPE


model = UNet2DConditionModel(
    sample_size=RESPONSE_SHAPE[0],
    in_channels=RESPONSE_SHAPE[-1] * 2,
    out_channels=RESPONSE_SHAPE[-1],
    layers_per_block=1,
    block_out_channels=(4, 8, 16, 32),
    down_block_types=(
        'DownBlock2D',
        'DownBlock2D',
        'CrossAttnDownBlock2D',
        'DownBlock2D',
    ),
    mid_block_type='UNetMidBlock2DCrossAttn',
    up_block_types=(
        'UpBlock2D',
        'CrossAttnUpBlock2D',
        'UpBlock2D',
        'UpBlock2D',
    ),
    encoder_hid_dim=PARTICLE_SHAPE[0],
    cross_attention_dim=16,
    attention_head_dim=4,
    norm_num_groups=4
)
