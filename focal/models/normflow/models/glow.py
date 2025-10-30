from nflows import flows, distributions, transforms
from nflows.nn import nets


import numpy as np
import torch
from torch import nn
from nflows.transforms.base import Transform

from focal.models.normflow.mixing import HouseholderMix, InvertibleConv1x1LU, Squeeze2x2Vec



def build_caloinn_glow_like_flow(
    H=106, W=106,                   # <- parzyste!
    context_features=3,
    hidden_features=256,
    num_layers=12,
    num_bins=10,
    num_blocks=4,
    tail_bound=3.0,
    use_actnorm=True,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    postCouplingTransform='RandomPermutation',  # 'RandomPermutation' | 'Householder' | 'Conv1x1'
    device="cpu"
):
    assert H % 2 == 0 and W % 2 == 0, "Użyj parzystych H,W (dopaduj 105->106), aby włączyć squeeze."
    C0 = 1
    D = C0 * H * W

    # Po jednorazowym squeeze: [C=4, H/2, W/2], ale D się nie zmienia
    C1, H1, W1 = C0 * 4, H // 2, W // 2

    # maska checkerboard w AFTER-SQUEEZE siatce, powielona na kanały
    yy, xx = np.indices((H1, W1))
    mask2d = ((yy + xx) % 2 == 0).astype("float32")      # [H1,W1]
    maskCHW = np.broadcast_to(mask2d, (C1, H1, W1))      # [C1,H1,W1]
    mask = torch.from_numpy(maskCHW.reshape(-1))         # [D], CPU

    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            activation=nn.ReLU(),
            use_batch_norm=False,
            dropout_probability=0.0,
        )

    chain = []
    # --- SQUEEZE 2x2 raz na wejściu ---
    chain.append(Squeeze2x2Vec(C=C0, H=H, W=W))   # D const

    for i in range(num_layers):
        if use_actnorm:
            chain.append(transforms.normalization.ActNorm(features=D))

        # Coupling RQS na wektorze po squeeze (mask computed for C1,H1,W1)
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

        # --- post-mixing ---
        if postCouplingTransform == 'RandomPermutation':
            chain.append(transforms.RandomPermutation(features=D))
        elif postCouplingTransform == 'Householder':
            chain.append(HouseholderMix(features=D, num_reflections=4))
        elif postCouplingTransform == 'Conv1x1':
            chain.append(InvertibleConv1x1LU(C=C1, H=H1, W=W1, use_permutation=True))
        else:
            raise ValueError(f"Unknown postCouplingTransform={postCouplingTransform}")

    transform = transforms.CompositeTransform(chain)
    base = distributions.StandardNormal(shape=[D])
    flow = flows.Flow(transform=transform, distribution=base).to(device)
    return flow, { 'flatten': False, 'pad': 1 }
