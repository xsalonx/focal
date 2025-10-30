import time
import os
from argparse import ArgumentParser
from collections import defaultdict

from pprint import pprint

import jax
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from tqdm.auto import trange, tqdm

from focal.utils.data import load_data_scope
from focal.utils.metrics import Metrics
from focal.utils.eval import eval_step

from focal.utils.train import get_last_epoch, get_pretrained_from_checkpoint, get_all_saved_epochs, get_pretrained_from_checkpoint_by_epoch

from pathlib import Path
from utilss.physical_loss import calc_positional_loss
from focal.models.diffusion.utils import remove_noise, add_dump_metrics, DDIMConditionPipeline

from focal.models.diffusion.models.model1 import model as model1
from focal.models.diffusion.models.model2 import model as model2
from focal.models.diffusion.models.model3 import model as model3


noise_scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=True, clip_sample_range=7.0, timestep_spacing='trailing')

def get_model(name):
    models_lookup = {
        1: (model1, ),
        2: (model2, ),
        3: (model3, ),
    }
    return models_lookup[name]


from diffusers import get_cosine_schedule_with_warmup

def get_optimizer(model, epochs, batch_size, n_examples, lr):
    """
        Get optimizer and lr_scheduler

        Return: 
            optimizer, lr_scheduler
    """
    steps = epochs * n_examples // batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.87, 0.82), eps=2.5e-10, weight_decay=5.5e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * steps, num_training_steps=steps)
    return optimizer, lr_scheduler




def train_loop(
        name,
        model,
        metrics: Metrics,
        noise_scheduler,
        optimizer,
        data_scope,
        lr_scheduler,
        device,
        epochs,

        generator=torch.manual_seed(42),
        n_rep_test=2,
        resume=True,
        checkpoints_dir="./checkpoints",
        training_params=None,
        positional_loss_type=None,
        positional_loss_lambda=None,

        val_freq = 2,
        **kwargs,
    ):
    
    os.makedirs(checkpoints_dir, exist_ok=True)

    if resume:
        last_epoch = get_last_epoch(checkpoints_dir)
        if last_epoch is not None:
            pretrained_pipeline = get_pretrained_from_checkpoint(checkpoints_dir, DDIMConditionPipeline)
            if pretrained_pipeline is not None:
                model = pretrained_pipeline.unet
        else:
            last_epoch = 0


    model.to(device)

    print("Number of parameters:", model.num_parameters())

    sample_particles, sample_responses, *_ = data_scope['samples']
    sample_particles = torch.Tensor(sample_particles).to(device)
    sample_responses = sample_responses.transpose(0, 2, 3, 1)

    for epoch in trange(last_epoch, epochs, desc=f'Epochs {name}'):
        def train_step_fn():
            model.train()
            for batch in tqdm(data_scope['dataloaders']['train'], desc="Train loader"):
                cond, responses, raw_cond = batch

                responses = responses.to(device)
                cond = cond.to(device)
                optimizer.zero_grad()

                noise = torch.randn(responses.shape, generator=generator).to(device)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (responses.shape[0],), generator=generator).to(device).long()

                noisy_images = noise_scheduler.add_noise(responses, noise, timesteps)
                noise_pred = model(noisy_images, timesteps, cond).sample
                noise_loss = F.mse_loss(noise_pred, noise)
                metric_values = { 'mse_noise': noise_loss.detach().item() }

                if positional_loss_type is not None and positional_loss_lambda is not None:
                    denoised_image = remove_noise(noise_scheduler, noisy_images, noise_pred, timesteps).squeeze()

                    positional_loss, comp_coords, coords = calc_positional_loss(
                        raw_cond,
                        denoised_image,
                        positional_loss_type)
                    
                    metric_values[f'positional_{positional_loss_type}'] = positional_loss.detach().item()

                    comp_coords = comp_coords.detach().cpu().numpy()
                    coords = coords.detach().cpu().numpy()

                    add_dump_metrics(metric_values, comp_coords, coords)

                    loss = noise_loss + positional_loss * positional_loss_lambda
                else:
                    loss = noise_loss


                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                metrics.add(metric_values, 'train', section='train')

        train_step_fn()


# VALIDATION
        if epoch % val_freq == 0:
            def eval_step_fn():
                model.eval()
                with torch.no_grad():
                    for num_inference_steps in [25, 50]: #100]: # 10 and 100
                        pipeline = DDIMConditionPipeline(unet=model, scheduler=noise_scheduler)
                        pipeline_f = lambda cond, *_: pipeline(cond.to(device), generator, num_inference_steps=num_inference_steps).images
                        metrics.current_epoch = epoch
                        eval_step(pipeline_f, data_scope['dataloaders']['test'], data_scope, 'validation', metrics, tag=f'i{num_inference_steps}')

                        generated_samples = pipeline(sample_particles, generator=torch.manual_seed(0), num_inference_steps=num_inference_steps).images
                        metrics.plot_responses(
                            [sample_responses, generated_samples],
                            epoch,
                            tag=f"i{num_inference_steps}",
                            labels=["Original", "Generated"],
                        )

                    pipeline.save_pretrained(checkpoints_dir / f'epoch_{epoch + 1}')


            eval_step_fn()

        metrics.log(epoch)


# TEST
    print(f'Testing after {epochs} epochs of trainig')
    with torch.no_grad():
        for num_inference_steps in [25, 50]: #100]: # 10 and 100
            pipeline = DDIMConditionPipeline(unet=model, scheduler=noise_scheduler)
            pipeline_f = lambda cond, *_: pipeline(cond.to(device), generator, num_inference_steps=num_inference_steps).images
            metrics.current_epoch = epoch
            eval_step(pipeline_f, data_scope['dataloaders']['test'], data_scope, 'test', metrics, tag=f'i{num_inference_steps}', n_rep=n_rep_test)

        metrics.log(epochs)
        print(f'Finished training of ${name}')

def reval_loop(
        name,
        model,
        metrics: Metrics,
        data_scope,
        device,
        checkpoints_dir,
        generator=torch.manual_seed(42),
    ):
    

    model.to(device)

    sample_particles, sample_responses, *_ = data_scope['samples']
    sample_particles = torch.Tensor(sample_particles).to(device)
    sample_responses = sample_responses.transpose(0, 2, 3, 1)
    from focal.utils.train import find_closest_epoch_file 
    saved_epochs = [find_closest_epoch_file(checkpoints_dir, args.ftest)]

    model.eval()
    with torch.no_grad():
        for epoch in tqdm(saved_epochs, desc=f'Epochs {name}'):
            pipeline = get_pretrained_from_checkpoint_by_epoch(checkpoints_dir, DDIMConditionPipeline, epoch=epoch).to(device)
            for num_inference_steps in [25, 50]: #100]: # 10 and 100
                pipeline_f = lambda cond, *_: pipeline(cond.to(device), generator, num_inference_steps=num_inference_steps).images
                metrics.current_epoch = epoch
                eval_step(pipeline_f, data_scope['dataloaders']['test'], data_scope, 'ftest', metrics, tag=f'i{num_inference_steps}')

            metrics.log(epoch, logAsEpoch=True)



def build_parser():
    args = ArgumentParser()

    args.add_argument("--name", "-n", required=True, type=str)
    args.add_argument("--model-id", "--mid", required=True, type=int)

    args.add_argument("--train-batch-size", "--tbs", required=True, type=int)
    args.add_argument("--eval-batch-size", "--ebs", default=1024, required=False, type=int)
    args.add_argument("--test-batch-size", "--tsbs", default=1024, required=False, type=int)

    args.add_argument("--epochs", "--ep", "-e", required=False, default=100, type=int)
    args.add_argument("--learning-rate", "--lr", default=2.8e-5, type=float)

    args.add_argument("--data-dir", "--dd", type=str, required=True)
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

    args.add_argument("--val-freq", "--valf", type=int, default=2, help="Validate every N epochs")

    args.add_argument("--responses-scaler", "--rssc", default="1log+1", type=str)

    args.add_argument("--ftest", default=None, type=int)

    return args


from focal.utils.formats import sci_e

def extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))
    return f"{kwargs['name']}-{kwargs['model_id']}-{kwargs['train_batch_size']}-{sci_e(kwargs['learning_rate'])}-{kwargs['positional_loss_type']}-{sci_e(kwargs['positional_loss_lambda'])}"

if __name__ == '__main__':
    args = build_parser().parse_args()
    pprint(vars(args))
    name = extend_name_with_params(**vars(args))
    print(name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_device = device if args.early_loaders_to_cuda else 'cpu'

    data_scope = load_data_scope(
        device=data_device,
        **vars(args),
        data_load_config=[
            ['particles', 'standard', lambda p: p[:, None]],
            ['responses', args.responses_scaler, lambda p: p[:, None]],
            ['particles', 'none', lambda p: p[:, None]],
        ]
    )

    checkpoints_dir = Path(args.trainings_dir).resolve() / name / 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    model, = get_model(args.model_id)
    optimizer, lr_scheduler = get_optimizer(model, epochs=args.epochs, batch_size=args.train_batch_size, n_examples=data_scope['n_examples']['train'], lr=args.learning_rate)

    config = vars(args)
    config['model_type'] = str(type(model))
    config['model_family'] = 'diffusion'
    from focal.utils.train import get_num_parameters
    config['model_num_parameters'] = get_num_parameters(model)

    metrics = Metrics(
        job_type='train',
        name=name,
        trainings_dir=args.trainings_dir,
        use_wandb=not args.no_wandb,
        config=config)

    if not args.ftest:
        print("Training")
        train_loop(
            model=model,
            metrics=metrics,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            data_scope=data_scope,
            checkpoints_dir=checkpoints_dir,
            training_params=config,
            **vars(args),
        )

    else:
        print("Revalidation")
        reval_loop(
            name = args.name,
            model=model,
            metrics=metrics,
            device=device,
            data_scope=data_scope,
            checkpoints_dir=checkpoints_dir,
        )