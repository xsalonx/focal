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
from utilss.image_analyzis.energy_characteristics import get_image_from_characteristics


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
    args.add_argument("--model-id", "--mid", required=True, type=int)
    args.add_argument("--latent-dim", type=int, default=16)
    args.add_argument("--learning-rate", "--lr", type=float)
    args.add_argument("--train-batch-size", "--tbs", required=True, type=int)

    args.add_argument("--enhancer-name", "-en", required=True, type=str)
    args.add_argument("--enhancer-g-model-id", "--egmid", required=True, type=int)
    args.add_argument("--enhancer-generator-learning-rate", "--eglr", type=float)
    args.add_argument("--enhancer-d-model-id", "--edmid", required=True, type=int)
    args.add_argument("--enhancer-discriminator-learning-rate", "--edlr", type=float)
    args.add_argument("--enhancer-train-batch-size", "--etbs", required=True, type=int)
    args.add_argument("--enhancer-latent-dim", type=int, default=32)

    args.add_argument("--eval-batch-size", "--ebs", default=1024, required=False, type=int)
    args.add_argument("--test-batch-size", "--tsbs", default=1024, required=False, type=int)

    args.add_argument("--epochs", "--ep", "-e", required=False, default=500, type=int)

    args.add_argument("--data-dir", "--dd", type=str)
    args.add_argument("--trainings-dir", "--td", type=str)

    args.add_argument("--early-loaders-to-cuda", "--elcuda", action="store_true", default=False)

    args.add_argument("--subset-percentage", "--subpct", type=float)
    args.add_argument("--samples-no", default=32, type=int)

    args.add_argument("--no-wandb", "--nowb", action="store_true", default=False)
    args.add_argument("--group", "--gr", required=False, type=str)

    args.add_argument("--n-rep-test", "--nrt", "-r", default=1, type=int)

    args.add_argument("--no-params-in-name", "--no-pin", action="store_true", default=False)

    args.add_argument("--beta1", type=float, default=0.9)
    args.add_argument("--val-freq", type=int, default=2, help="Validate every N epochs")

    args.add_argument("--no-resume", default=False, action="store_true")

    args.add_argument("--responses-scaler", "--rssc", default="1log+1", type=str)

    args.add_argument("--ftest", default=None, type=int)

    return args


from focal.utils.formats import sci_e

from focal.models.approx.gan.generators import GeneratorUpsamplingBigger, GeneratorUpsamplingBiggerLessChannels, GeneratorUpsamplingSmallerNoRes
from focal.models.approx.gan.discriminators import Discriminator

def get_generator(id):
    models_lookup = {
        2: GeneratorUpsamplingBigger,
        3: GeneratorUpsamplingBiggerLessChannels,
        5: GeneratorUpsamplingSmallerNoRes,
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

def approx_extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))
    return f"{kwargs['name']}-{kwargs['model_id']}-{kwargs['latent_dim']}-{kwargs['train_batch_size']}-{sci_e(kwargs['learning_rate'])}"


def enhancer_extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))
    return f"{kwargs['enhancer_name']}-{kwargs['enhancer_g_model_id']}-{kwargs['enhancer_d_model_id']}-{kwargs['enhancer_train_batch_size']}-{sci_e(kwargs['enhancer_generator_learning_rate'])}-{sci_e(kwargs['enhancer_discriminator_learning_rate'])}"


from focal.models.gan.utils import save_gan, load_gan
from focal.utils.train import get_last_epoch
from pathlib import Path
import os

if __name__ == '__main__':
    args = build_parser().parse_args()
    pprint(vars(args))
    enhancer_name = enhancer_extend_name_with_params(**vars(args))
    print(enhancer_name)

    # 1️⃣  Data ----------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_device = device if args.early_loaders_to_cuda else 'cpu'
    
    generator=torch.manual_seed(42)

    apporx_name = approx_extend_name_with_params(**vars(args))
    print(apporx_name)

    data_scope_args = {**vars(args)}
    data_scope_args['train_batch_size'] = args.enhancer_train_batch_size
    data_scope = load_data_scope(
        device=data_device,
        **data_scope_args,
        data_load_config=[
            ['particles', 'standard'],
            ['responses', args.responses_scaler, lambda p: p[:, None]],
            [f'approx/{apporx_name}/approx', 'none', lambda p: p[:, None]]
        ]
    )


    enhancer_checkpoints_dir = Path(args.trainings_dir).resolve() / enhancer_name / 'checkpoints'
    os.makedirs(enhancer_checkpoints_dir, exist_ok=True)
    
# Loading Enhancer
    COND_DIM=3
    generator_model = get_generator(args.enhancer_g_model_id)(args.enhancer_latent_dim, COND_DIM).to(device)
    response_shape = data_scope['dataslices']['train'][1][0].shape
    discriminator_model = get_discriminator(args.enhancer_d_model_id)(response_shape, COND_DIM).to(device)
    generator_optimizer = _make_optimizer(generator_model.parameters(), args.enhancer_generator_learning_rate, args.beta1)
    discriminator_optimizer = _make_optimizer(discriminator_model.parameters(), args.enhancer_discriminator_learning_rate, args.beta1)
    if not args.no_resume:
        last_epoch = get_last_epoch(enhancer_checkpoints_dir)
        if last_epoch is not None:
            load_gan(enhancer_checkpoints_dir, generator_model, discriminator_model, last_epoch, generator_optimizer, discriminator_optimizer)
            print(f"Loaded state from checkpoint after epoch {last_epoch}")
        else:
            last_epoch = 0

    config = {**vars(args)}
    config['model_type'] = str(type(generator_model)) + str(type(discriminator_model))
    config['model_family'] = 'gan-enhancer'
    config['name'] = args.enhancer_name
    config['approx_name'] = apporx_name
    from focal.utils.train import get_num_parameters
    config['model_num_parameters'] = get_num_parameters(generator_model)
    config['model_discriminator_num_parameters'] = get_num_parameters(discriminator_model)
    
    metrics = Metrics(
        job_type='train',
        name=enhancer_name,
        trainings_dir=args.trainings_dir,
        use_wandb=not args.no_wandb,
        config=config)


    sample_particles, sample_responses, sample_approx = data_scope['samples']
    sample_particles = torch.Tensor(sample_particles).to(device)
    sample_responses = sample_responses.transpose(0, 2, 3, 1)
    sample_approx = torch.Tensor(sample_approx).to(device)
    sample_approx_image = sample_approx.cpu().numpy().transpose(0, 2, 3, 1)
    
    
    def generation_pipeline(*batch):
        cond = batch[0]
        approx = batch[2]
        z = torch.randn(cond.size(0), args.enhancer_latent_dim, device=device)
        return generator_model(z, cond.to(device), approx.to(device))
    
    import numpy as np
    eval_pipeline = lambda *batch: np.maximum(generation_pipeline(*batch).permute(0, 2, 3, 1).cpu(), 0)

    # 3️⃣  Training ------------------------------------------------------------

    if not args.ftest:
        for epoch in range(last_epoch + 1, args.epochs):
            print(f"Epoch {epoch}")
            generator_model.train()
            discriminator_model.train()
            for batch_idx, batch in enumerate(tqdm(data_scope['dataloaders']['train'], desc="Train loader")):
                generator_optimizer.zero_grad()
                discriminator_optimizer.zero_grad()

                cond, responses, *other_data = batch
                responses = responses.to(device)
                cond = cond.squeeze().to(device)
                batch_size = responses.size(0)
                if batch_size == 1:
                    continue

                # ------------------ G forward ------------------
                fake_responses = generation_pipeline(*batch)
                # D pass
                reals_discrimnation = discriminator_model(responses, cond)
                fakes_discrimnation = discriminator_model(fake_responses.detach(), cond)

                d_loss = discriminator_loss(reals_discrimnation, fakes_discrimnation)
                d_loss.backward()
                discriminator_optimizer.step()

                # G update
                fakes_discrimination = discriminator_model(fake_responses, cond)
                g_loss = generator_loss(fakes_discrimination)
                g_loss.backward()
                generator_optimizer.step()

                metric_values = {
                    'gen_loss': g_loss.detach().item(),
                    'disc_loss': d_loss.detach().item(),
                }
                metrics.add(metric_values, 'train', section='train')


            generator_model.eval()
            discriminator_model.eval()
            if epoch % args.val_freq == 0:
                with torch.no_grad():
                    metrics.current_epoch = epoch
                    eval_step(
                        eval_pipeline,
                        data_scope['dataloaders']['test'],
                        data_scope,
                        'validation',
                        metrics,
                        calc_perceptual_loss=epoch > 100,
                    )

                    z = torch.randn(sample_particles.size(0), args.enhancer_latent_dim, generator=generator).to(device)
                    generated_samples = generator_model(z, sample_particles, sample_approx).permute(0, 2, 3, 1).cpu().numpy()
                    metrics.plot_responses(
                            [sample_responses, generated_samples],
                            epoch,
                            labels=["Original", "Generated"],
                        )
            save_gan(enhancer_checkpoints_dir, generator_model, discriminator_model, epoch, generator_optimizer, discriminator_optimizer)

            metrics.log(epoch)

        print(f'Testing after {args.epochs} epochs of trainig')
        with torch.no_grad():
            metrics.current_epoch = epoch
            eval_step(
                    eval_pipeline,
                    data_scope['dataloaders']['test'],
                    data_scope,
                    'test',
                    metrics
            )
            metrics.log(args.epochs)
            print(f'Finished training of {enhancer_name}')

        print("Training finished ✔️")

    else:
        from focal.utils.train import get_all_saved_epochs
        from focal.utils.train import find_closest_epoch_file 
        saved_epochs = [find_closest_epoch_file(enhancer_checkpoints_dir, args.ftest)]
        generator_model.eval()
        discriminator_model.eval()
        with torch.no_grad():
            for epoch in tqdm(saved_epochs, desc=f'Epochs {enhancer_name}'):
                load_gan(enhancer_checkpoints_dir, generator_model, discriminator_model, epoch)
                metrics.current_epoch = epoch
                eval_step(
                    eval_pipeline,
                    data_scope['dataloaders']['test'],
                    data_scope,
                    'ftest',
                    metrics
                )
                metrics.log(epoch, logAsEpoch=True)
            
