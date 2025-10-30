from argparse import ArgumentParser
from collections import defaultdict

import jax
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from focal.utils.data import load_data_scope
from focal.utils.metrics import Metrics
from focal.utils.eval import eval_step

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_optimizer(params, lr: float, beta1: float = 0.5):
    """Adam (β2 kept 0.999 to mirror TF/Keras default)."""
    return torch.optim.Adam(params, lr=lr, betas=(beta1, 0.999))


# -----------------------------------------------------------------------------
# 🔥  Losses & Metrics
# -----------------------------------------------------------------------------


bce_logits = nn.BCEWithLogitsLoss()


def discriminator_loss(real_out, fake_out):
    real_targets = torch.ones_like(real_out)
    fake_targets = torch.zeros_like(fake_out)

    real_loss = bce_logits(real_out, real_targets)
    fake_loss = bce_logits(fake_out, fake_targets)
    loss = real_loss + fake_loss

    return loss


def generator_loss(fake_out):
    targets = torch.ones_like(fake_out)
    loss = bce_logits(fake_out, targets)
    return loss



# -----------------------------------------------------------------------------
# 🏃‍♀️  Main training script (CLI)
# -----------------------------------------------------------------------------

from pprint import pprint
from tqdm import tqdm

def build_parser():
    args = ArgumentParser()

    args.add_argument("--name", "-n", required=True, type=str)
    args.add_argument("--g-model-id", "--gmid", required=True, type=int)
    args.add_argument("--generator-learning-rate", "--glr", type=float)
    args.add_argument("--d-model-id", "--dmid", required=True, type=int)
    args.add_argument("--discriminator-learning-rate", "--dlr", type=float)

    args.add_argument("--train-batch-size", "--tbs", required=True, type=int)
    args.add_argument("--eval-batch-size", "--ebs", default=1024, required=False, type=int)
    args.add_argument("--test-batch-size", "--tsbs", default=1024, required=False, type=int)

    args.add_argument("--epochs", "--ep", "-e", required=False, default=500, type=int)

    args.add_argument("--data-dir", "--dd", type=str)
    args.add_argument("--trainings-dir", "--td", type=str)

    args.add_argument("--early-loaders-to-cuda", "--elcuda", action="store_true", default=False)

    args.add_argument("--subset-percentage", "--subpct", type=float)
    args.add_argument("--samples-no", default=32, type=int)

    args.add_argument("--positional-loss-type", "--pltype", type=str, choices=["L1", "L2"], required=False)
    args.add_argument("--positional-loss-lambda", "--pllambda", type=float, required=False)

    args.add_argument("--no-wandb", "--nowb", action="store_true", default=False)
    args.add_argument("--group", "--gr", required=False, type=str)

    args.add_argument("--n-rep-test", "--nrt", "-r", default=1, type=int)

    args.add_argument("--no-params-in-name", "--no-pin", action="store_true", default=False)

    args.add_argument("--latent-dim", type=int, default=32)
    args.add_argument("--beta1", type=float, default=0.9)
    args.add_argument("--val-freq", type=int, default=2, help="Validate every N epochs")

    args.add_argument("--no-resume", default=False, action="store_true")

    args.add_argument("--responses-scaler", "--rssc", default="1log+1", type=str)

    args.add_argument("--ftest", default=None, type=int)

    return args


from focal.utils.formats import sci_e

def extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))
    return f"{kwargs['name']}-{kwargs['g_model_id']}-{kwargs['d_model_id']}-{kwargs['train_batch_size']}-{sci_e(kwargs['generator_learning_rate'])}-{sci_e(kwargs['discriminator_learning_rate'])}-{kwargs['positional_loss_type']}-{sci_e(kwargs['positional_loss_lambda'])}"


from focal.models.gan.generators import GeneratorUpsamplingBigger, GeneratorUpsamplingSmaller, GeneratorTranConv1
from focal.models.gan.generators import GeneratorUpsamplingNoRes, GeneratorUpsamplingSmallerNoRes
from focal.models.gan.generators_v2.generator_1 import GeneratorUpsamplingBigger_v2_1
from focal.models.gan.generators_v2.generator_2 import GeneratorUpsamplingBigger_v2_2
from focal.models.gan.discriminators import Discriminator

def get_generator(id):
    models_lookup = {
        1: GeneratorUpsamplingSmaller,
        2: GeneratorUpsamplingBigger,
        3: GeneratorTranConv1,
        4: GeneratorUpsamplingNoRes,
        5: GeneratorUpsamplingSmallerNoRes,

        21: GeneratorUpsamplingBigger_v2_1,
        22: GeneratorUpsamplingBigger_v2_2,
    }
    return models_lookup[id]

def get_discriminator(id):
    models_lookup = {
        1: Discriminator,
    }
    return models_lookup[id]

from focal.models.gan.utils import save_gan, load_gan
from focal.utils.train import get_last_epoch
from pathlib import Path
from utilss.physical_loss import calc_positional_loss

if __name__ == '__main__':
    args = build_parser().parse_args()
    pprint(vars(args))
    name = extend_name_with_params(**vars(args))
    print(name)
    checkpoints_dir = Path(args.trainings_dir).resolve() / name / 'checkpoints'

    # 1️⃣  Data ----------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_device = device if args.early_loaders_to_cuda else 'cpu'
    
    generator=torch.manual_seed(42)

    data_scope = load_data_scope(
        device=data_device,
        **vars(args),
        data_load_config=[
            ['particles', 'standard'],
            ['responses', args.responses_scaler, lambda p: p[:, None]],
            ['particles', 'none'],
        ]
    )


    # 2️⃣  Models --------------------------------------------------------------
    COND_DIM=3
    generator_model = get_generator(args.g_model_id)(args.latent_dim, COND_DIM).to(device)
    response_shape = data_scope['dataslices']['train'][1][0].shape
    discriminator_model = get_discriminator(args.d_model_id)(response_shape, COND_DIM).to(device)
    generator_optimizer = _make_optimizer(generator_model.parameters(), args.generator_learning_rate, args.beta1)
    discriminator_optimizer = _make_optimizer(discriminator_model.parameters(), args.discriminator_learning_rate, args.beta1)
    if not args.no_resume:
        last_epoch = get_last_epoch(checkpoints_dir)
        if last_epoch is not None:
            load_gan(checkpoints_dir, generator_model, discriminator_model, last_epoch, generator_optimizer, discriminator_optimizer)
            print(f"Loaded state from checkpoint after epoch {last_epoch}")
        else:
            last_epoch = 0

    config = vars(args)
    config['model_type'] = str(type(generator_model)) + str(type(discriminator_model))
    config['model_family'] = 'gan'
    from focal.utils.train import get_num_parameters
    config['model_num_parameters'] = get_num_parameters(generator_model)
    config['model_discriminator_num_parameters'] = get_num_parameters(discriminator_model)
    
    print(name)
    metrics = Metrics(
        job_type='train',
        name=name,
        trainings_dir=args.trainings_dir,
        use_wandb=not args.no_wandb,
        config=config)
    

    sample_particles, sample_responses, *_ = data_scope['samples']
    sample_particles = torch.Tensor(sample_particles).to(device)
    sample_responses = sample_responses.transpose(0, 2, 3, 1)

    import numpy as np

    pipeline = lambda cond, *_: np.maximum(generator_model(torch.randn(cond.size(0), args.latent_dim, device=device), cond.to(device)).permute(0, 2, 3, 1).cpu(), 0)
    
    # 3️⃣  Training ------------------------------------------------------------
    if not args.ftest:
        for epoch in range(last_epoch + 1, args.epochs):
            print(f"Epoch {epoch}")
            generator_model.train()
            discriminator_model.train()
            for batch_idx, batch in enumerate(tqdm(data_scope['dataloaders']['train'], desc="Train loader")):
                generator_optimizer.zero_grad()
                discriminator_optimizer.zero_grad()

                cond, responses, raw_cond = batch
                responses = responses.to(device)
                cond = cond.to(device)
                batch_size = responses.size(0)
                if batch_size == 1:
                    continue

                z = torch.randn(batch_size, args.latent_dim, device=device)

                # ------------------ G forward ------------------
                fake_responses = generator_model(z, cond)
                # D pass
                reals_discrimnation = discriminator_model(responses, cond)
                fakes_discrimnation = discriminator_model(fake_responses.detach(), cond)

                d_loss = discriminator_loss(reals_discrimnation, fakes_discrimnation)
                d_loss.backward()
                discriminator_optimizer.step()

                # G update
                fakes_discrimination = discriminator_model(fake_responses, cond)
                g_loss = generator_loss(fakes_discrimination)


                metric_values = {}
                if args.positional_loss_type is not None and args.positional_loss_lambda is not None:
                    positional_loss, _, __ = calc_positional_loss(
                        raw_cond,
                        fake_responses.squeeze(1),
                        args.positional_loss_type)
                    
                    metric_values[f'positional_{args.positional_loss_type}'] = positional_loss.detach().item()
                    g_loss = g_loss + positional_loss * args.positional_loss_lambda
                    


                g_loss.backward()
                generator_optimizer.step()

                metric_values['gen_loss'] = g_loss.detach().item()
                metric_values['disc_loss'] = d_loss.detach().item()

                metrics.add(metric_values, 'train', section='train')


            generator_model.eval()
            discriminator_model.eval()
            if epoch % args.val_freq == 0:
                with torch.no_grad():
                    metrics.current_epoch = epoch
                    eval_step(
                        pipeline,
                        data_scope['dataloaders']['test'],
                        data_scope,
                        'validation',
                        metrics,
                        calc_perceptual_loss=epoch > args.epochs / 2,
                    )

                    z = torch.randn(sample_particles.size(0), args.latent_dim, generator=generator).to(device)
                    generated_samples = generator_model(z, sample_particles).permute(0, 2, 3, 1).cpu().numpy()
                    metrics.plot_responses(
                            [sample_responses, generated_samples],
                            epoch,
                            labels=["Original", "Generated"],
                        )
            save_gan(checkpoints_dir, generator_model, discriminator_model, epoch, generator_optimizer, discriminator_optimizer)

            metrics.log(epoch)

        print(f'Testing after {args.epochs} epochs of trainig')
        with torch.no_grad():
            metrics.current_epoch = epoch
            eval_step(pipeline, data_scope['dataloaders']['test'], data_scope, 'test', metrics)
            metrics.log(args.epochs)
            print(f'Finished training of {name}')

        print("Training finished ✔️")


    else:
        from focal.utils.train import get_all_saved_epochs
        from focal.utils.train import find_closest_epoch_file 
        saved_epochs = [find_closest_epoch_file(checkpoints_dir, args.ftest)]
        generator_model.eval()
        discriminator_model.eval()
        with torch.no_grad():
            for epoch in tqdm(saved_epochs, desc=f'Epochs {name}'):
                load_gan(checkpoints_dir, generator_model, discriminator_model, epoch)
                metrics.current_epoch = epoch
                eval_step(
                    pipeline,
                    data_scope['dataloaders']['test'],
                    data_scope,
                    'ftest',
                    metrics
                )
                metrics.log(epoch, logAsEpoch=True)
            
