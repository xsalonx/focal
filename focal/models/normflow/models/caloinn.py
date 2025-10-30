

import math, numpy as np, torch
from torch import nn
from nflows import flows, distributions, transforms
from nflows.nn import nets
import math, numpy as np, torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from nflows import flows, distributions, transforms
from nflows.nn import nets

from focal.models.normflow.mixing import HouseholderMix


def build_caloinn_like_flow(
    hidden_features,
    num_layers,
    num_bins,
    num_blocks,
    tail_bound,
    use_actnorm,
    min_bin_width,
    min_bin_height,
    min_derivative,
    postCouplingTransform,
    context_features=3,
    device="cpu",
    H=105, W=105,
):
    D = H * W

    # checkerboard mask (CPU!)
    yy, xx = np.indices((H, W))
    mask2d = ((yy + xx) % 2 == 0).astype("float32")
    mask = torch.from_numpy(mask2d.reshape(-1))  # [D], CPU

    def create_net(in_features, out_features):
        # MLP (ResidualNet) z kontekstem log(E_inc)
        return nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,    
            activation=nn.ReLU(),
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    chain = []
    for i in range(num_layers):
        if use_actnorm:
            chain.append(transforms.normalization.ActNorm(features=D))
        m = mask if (i % 2 == 0) else (1.0 - mask)
        chain.append(
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=m,
                transform_net_create_fn=create_net,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        )
        if postCouplingTransform == 'RandomPermutation':
            chain.append(transforms.RandomPermutation(features=D))
        elif postCouplingTransform == 'LULinear':
            chain.append(transforms.LULinear(features=D))
        elif postCouplingTransform == 'Householder':
            chain.append(HouseholderMix(features=D, num_reflections=4))


    transform = transforms.CompositeTransform(chain)
    base = distributions.StandardNormal(shape=[D])
    flow = flows.Flow(transform=transform, distribution=base).to(device)

    return flow, { 'flatten': True, 'pad': 0}
