# %%

PARTICLE_SHAPE = (3,)
RESPONSE_SHAPE = (105, 105)

import torch
import sys

sys.path.append('../../..')

from focal.utils.data import *

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from utilss.image_analyzis.energy_characteristics import get_image_from_characteristics
from argparse import ArgumentParser
from focal.utils.metrics import Metrics
from focal.utils.train import load_ckpt, save_ckpt

import os

from focal.models.normflow.models.caloinn import build_caloinn_like_flow
from focal.models.normflow.models.glow import build_caloinn_glow_like_flow


import math, numpy as np, torch
import math, numpy as np, torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR


def build_parser():
    args = ArgumentParser()

    args.add_argument("--name", "-n", required=True, type=str)
    args.add_argument("--model", "-m", required=False, default='caloinn', choices=['caloinn', 'caloinn-glow'])

    # For 30_000 samples
    # 32 85
    # 64 170
    # 128 340
    #
    #
    args.add_argument("--train-batch-size", "--tbs", default=32, type=int)
    args.add_argument("--epochs", "--e", default=256, type=int)
    args.add_argument("--test-batch-size", default=32, type=int)
    args.add_argument("--learning-rate", "--lr", default=1e-4, type=float)

    args.add_argument("--data-dir", "--dd", type=str, default="data/single-shower-first/cleaned-1", required=False)

    args.add_argument("--hidden-features", "--hf", default=128, type=int)
    args.add_argument("--num-layers", "--nl", default=3, type=int)
    args.add_argument("--num-bins", "--nb", default=4, type=int)
    args.add_argument("--num-blocks", "--nbl", default=4, type=int)
    args.add_argument('--min-bin-width', type=float, default=1e-4)
    args.add_argument('--min-bin-height', type=float, default=1e-4)
    args.add_argument('--min-derivative', type=float, default=1e-4)
    args.add_argument('--tail-bound', type=float, default=7)
    args.add_argument('--post-coupling-transform', "--pct", default='Conv1x1', choices=['RandomPermutation', 'Householder',  'Conv1x1'], type=str)

    args.add_argument("--responses-scaler", default='caloinn', type=str)
    args.add_argument("--subset-percentage", "--subpct", default=1, type=float)
    args.add_argument("--context-features", "--cf", default=3, type=int)

    args.add_argument("--no-actnorm", "--nac", default=False, action='store_true')


    args.add_argument("--val-freq", "--valf", type=int, default=2, help="Validate every N epochs")

    args.add_argument("--omit-in-name", "--omitn", type=str, default='')
    args.add_argument("--trainings-dir", "--td", type=str, default='./trainings')

    args.add_argument("--ftest", type=int)

    args.add_argument("--force-name", type=str)

    return args


from focal.utils.formats import sci_e
from collections import defaultdict

def extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))

    ommit = kwargs['omit_in_name'].split(',') if kwargs['omit_in_name'] else []
  
    parts = [
        str(kwargs['name']),
        str(kwargs['model']),
        str(kwargs['train_batch_size']),
        str(kwargs['epochs']),
        str(kwargs['learning_rate']) if 'learning_rate' not in ommit else None,
        str(sci_e(kwargs['hidden_features'])),
        str(kwargs['num_layers']),
        str(kwargs['num_bins']),
        str(kwargs['num_blocks']),
        str(sci_e(kwargs['min_bin_width'])),
        str(sci_e(kwargs['min_bin_height'])),
        str(sci_e(kwargs['min_derivative'])),
        str(sci_e(kwargs['tail_bound'])),
        str(kwargs['post_coupling_transform']),
    ]

    valid_parts = [p for p in parts if p is not None]

    suffix = "XX"
    return "-".join(valid_parts) + suffix



from focal.utils.eval import eval_step
from focal.utils.train import get_slurm_seconds_left_secs

from pprint import pprint
if __name__ == '__main__':

    args = build_parser().parse_args()

    model_id = 0
    eval_batch_size = 1024
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    data_dir = args.data_dir
    trainings_dir = args.trainings_dir
    val_freq = args.val_freq
    responses_scaler_name = args.responses_scaler

    no_wandb = False

    config = {
        'name': args.name,
        'model_id': model_id,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': eval_batch_size,
        'test_batch_size': test_batch_size,
        'epochs': epochs,
        'learning_rate': args.learning_rate,
        'data_dir': data_dir,
        'trainings_dir': trainings_dir,
        'no_wandb': no_wandb,
        'val_freq': val_freq,
        'responses_scaler': responses_scaler_name,
    }

    name = extend_name_with_params(**vars(args))
    if args.force_name:
        name = args.force_name

    learning_rate = args.learning_rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'caloinn':
        flow, train_spec = build_caloinn_like_flow(
            hidden_features=args.hidden_features,
            num_layers=args.num_layers,
            num_bins=args.num_bins,
            context_features=args.context_features,
            num_blocks=args.num_blocks,
            postCouplingTransform=args.post_coupling_transform,
            min_bin_width=args.min_bin_width,
            min_bin_height=args.min_bin_height,
            min_derivative=args.min_derivative,
            use_actnorm=not args.no_actnorm,
            tail_bound=args.tail_bound,
        )
        H, W = 105, 105
    elif args.model == 'caloinn-glow':
        flow, train_spec = build_caloinn_glow_like_flow(
            hidden_features=args.hidden_features,
            num_layers=args.num_layers,
            num_bins=args.num_bins,
            context_features=args.context_features,
            num_blocks=args.num_blocks,
            min_bin_width=args.min_bin_width,
            min_bin_height=args.min_bin_height,
            min_derivative=args.min_derivative,
            tail_bound=args.tail_bound,
            postCouplingTransform=args.post_coupling_transform,
            use_actnorm=not args.no_actnorm,
        )
        H, W = 106, 106

    use_flatten = train_spec['flatten']
    pad = train_spec['pad']


    data_scope = load_data_scope(
        data_dir=data_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=eval_batch_size,
        test_batch_size=test_batch_size,
        samples_no=32,
        data_load_config=[
            ['particles', 'caloinnparticles'],
            ['responses', responses_scaler_name, lambda u: np.pad(u[:, None] if u.ndim == 3 else u, ((0, 0), (0, 0), (0, pad), (0, pad)))],
            # ['responses', responses_scaler],
        ],
        subset_percentage=args.subset_percentage,
    )


    responses_scaler = data_scope['scalers'][1]

    sample_particles, sample_responses, *_ = data_scope['samples']
    sample_particles = torch.Tensor(sample_particles).to(device)
    sample_images = sample_responses.transpose(0, 2, 3, 1)

    checkpoints_dir = Path(trainings_dir).resolve() / name / 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)


    flow = flow.to(device)

    opt = Adam(flow.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    steps_per_epoch = len(data_scope['dataloaders']['train'])
    sched = OneCycleLR(opt, max_lr=learning_rate, epochs=epochs, steps_per_epoch=steps_per_epoch)

    from focal.utils.train import get_num_parameters
    config['model_num_parameters'] = get_num_parameters(flow)
    config['model_type'] = str(type(flow))
    config['model_family'] = 'normflow'
    config = {**config, **vars(args)}
    pprint(config)

    metrics = Metrics(
            job_type='train',
            name=name,
            trainings_dir=trainings_dir,
            use_wandb=not no_wandb,
            config=config)

    try:
        print("CHK DIR", checkpoints_dir)
        if args.ftest:
            last_ep, _ = load_ckpt(checkpoints_dir, flow, epoch=None, map_location=device, strict=True)
        else:
            last_ep, _ = load_ckpt(checkpoints_dir, flow, optimizer=opt, scheduler=sched,
                            epoch=None, map_location=device, strict=True)
        start_epoch = last_ep + 1
        print(f"Resumed from epoch {last_ep}")
    except FileNotFoundError:
        start_epoch = 1
        print("Starting fresh.")

    grad_clip = 1

    # -------------------------
    # 4) Sampling: jedna próbka na kontekst
    # -------------------------
    @torch.no_grad()
    def sample_flow_caloinn(flow, cond, temperature=1.0, device="cuda"):
        """
        E_inc: [B] lub [B,1] (energia incident; jeśli cond_is_log=True -> już log(E_inc)).
        Zwraca: [B,1,H,W] w tej samej, „logowej” domenie co trening (odwróć swój preprocessing offline).
        """
        flow.eval()
        cond = cond.to(device=device, dtype=torch.float32)

        # # 1) wylosuj bazowy hałas z N(0, I) dla 1 próby na kontekst
        # z = flow._distribution.sample(1, context=cond)          # [1, B, D] (typowa forma w nflows)
        # # 2) temperatura: N(0, T^2 I) = T * N(0, I)
        # z = z.squeeze(1) * float(temperature)

        # # 3) odwrócenie transformacji (latent -> dane)
        # z, _ = flow._transform.inverse(z, context=cond)   

        # if z.dim() <= 3:
        #     z = z.reshape((z.shape[0], 1, H, W))
        # return z.cpu()
        
        z = flow.sample(1, context=cond.to(device))
        if z.dim() <= 3:
            z = z.reshape((z.shape[0], 1, H, W))
        return z.cpu()

    @torch.no_grad()
    def eval_pipeline(*batch, temperature=1.0):
        samples = sample_flow_caloinn(flow, batch[0], temperature=temperature)
        return samples.permute(0, 2, 3, 1).cpu()


    if args.ftest is None:
        epoch = None
        for epoch in range(start_epoch, epochs):
            tot, dims = 0.0, 0
            flow.train()
            pbar = tqdm(data_scope['dataloaders']['train'])
            for cond, imgs in pbar:
                imgs = imgs.to(device=device, dtype=torch.float32)         # [B,1,H,W]
                x = imgs
                x = imgs.reshape((x.shape[0], -1)) if use_flatten else x

                cond = cond[:, :args.context_features].to(device=device, dtype=torch.float32)

                nll = -flow.log_prob(x, context=cond).mean()

                opt.zero_grad(set_to_none=True)
                nll.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
                opt.step()
                sched.step()

                tot += nll.item() * x.size(0); dims += x.size(0)*x.size(1)
                bpd = (tot/dims)/math.log(2.0)
                pbar.set_description(f"Epoch {epoch:03d} | train bpd={bpd:.3f}")

        

            save_ckpt(checkpoints_dir, flow, optimizer=opt, scheduler=sched, epoch=epoch, max_to_keep=2)

            metrics.add({ 'bpd_loss': bpd }, section='train')
            metrics.log(epoch)
            
            if np.isnan(bpd):
                raise RuntimeError("bpd is NAN")

            if epoch % val_freq == 0:
                flow.eval()
                metrics.current_epoch = epoch
                try:
                    temp = 1
                    eval_step(
                        lambda *batch: eval_pipeline(*batch, temperature=temp),
                        data_scope['dataloaders']['val'],
                        data_scope,
                        'validation',
                        metrics,
                        tag=f"temp{sci_e(temp)}",
                        no_energy_dist_plot=False,
                        calc_perceptual_loss=False,
                        silent=False,
                    )
                except Exception as e:
                    pprint(e)

            temp = 1
            generated_samples = eval_pipeline(sample_particles, temperature=temp)
            metrics.plot_responses(
                [np.log1p(np.log1p(responses_scaler.inverse_transform(sample_images))), np.log1p(np.log1p(responses_scaler.inverse_transform(generated_samples)))],
                epoch,
                labels=["Original", "Generated"],
                tag=f"temp{sci_e(temp)}",
            )


            metrics.log(epoch, silent=True)

            time_left = get_slurm_seconds_left_secs()
            FTEST_TIME_MIN=15
            if time_left is not None:
                print(f"SLURM time left {time_left / 60} min [{FTEST_TIME_MIN}]")
                if time_left < FTEST_TIME_MIN * 60:
                    print("Existing final loop to do ftest")
                    break
    
    else:
        print("Straing FTEST")
        metrics.current_epoch = last_ep
        eval_step(
            eval_pipeline,
            data_scope['dataloaders']['test'],
            data_scope,
            'ftest',
            metrics,
        )
        metrics.log(last_ep)





