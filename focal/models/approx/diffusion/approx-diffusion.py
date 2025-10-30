from argparse import ArgumentParser

import numpy as np
import os
from pathlib import Path
from tqdm.auto import trange, tqdm

import torch
import torch.nn.functional as F

from focal.utils.metrics import Metrics
from focal.utils.train import get_last_epoch
from utilss.image_analyzis.energy_characteristics import get_image_from_characteristics


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

from focal.utils.train import get_last_epoch

from pathlib import Path


from focal.models.approx.diffusion.model1 import model as enhancerModel1
from focal.models.approx.diffusion.model2 import model as enhancerModel2
from focal.models.approx.diffusion.model3 import model as enhancerModel3

def get_enhancer_model_and_optimizer(name):
    models_lookup = {
        1: enhancerModel1,
        2: enhancerModel2,
        3: enhancerModel3
    }

    return models_lookup[name]


noise_scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=True, clip_sample_range=7.0, timestep_spacing='trailing')

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



from focal.models.approx.utils import *

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
    args.add_argument("--enhancer-model-id", "--emid", required=True, type=int)
    args.add_argument("--enhancer-learning-rate", "--elr", type=float)
    args.add_argument("--enhancer-train-batch-size", "--etbs", required=True, type=int)

    args.add_argument("--eval-batch-size", "--ebs", default=1024, required=False, type=int)
    args.add_argument("--test-batch-size", "--tsbs", default=1024, required=False, type=int)

    args.add_argument("--epochs", "--ep", "-e", required=False, default=100, type=int)

    args.add_argument("--data-dir", "--dd", type=str)
    args.add_argument("--trainings-dir", "--td", type=str)

    args.add_argument("--early-loaders-to-cuda", "--elcuda", action="store_true", default=False)

    args.add_argument("--subset-percentage", "--subpct", type=float)
    args.add_argument("--samples-no", default=32, type=int)

    args.add_argument("--no-wandb", "--nowb", action="store_true", default=False)
    args.add_argument("--group", "--gr", required=False, type=str)

    args.add_argument("--n-rep-test", "--nrt", "-r", default=1, type=int)

    args.add_argument("--beta1", type=float, default=0.9)
    args.add_argument("--val-freq", type=int, default=2, help="Validate every N epochs")

    args.add_argument("--no-resume", default=False, action="store_true")

    args.add_argument("--responses-scaler", "--rssc", default="1log+1", type=str)
    args.add_argument("--ftest", default=None, type=int)

    return args


from focal.utils.formats import sci_e
from collections import defaultdict

def approx_extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))
    return f"{kwargs['name']}-{kwargs['model_id']}-{kwargs['latent_dim']}-{kwargs['train_batch_size']}-{sci_e(kwargs['learning_rate'])}"

def enhancer_extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))
    return f"{kwargs['enhancer_name']}-{kwargs['enhancer_model_id']}-{kwargs['enhancer_base_channel']}-{kwargs['enhancer_train_batch_size']}-{sci_e(kwargs['enhancer_learning_rate'])}"




from focal.utils.eval import eval_step
from focal.utils.data import load_data_scope

import torch
from diffusers import DDIMPipeline, ImagePipelineOutput

from focal.utils.torch import torch_tensor_to_numpy_image
from diffusers.utils.torch_utils import randn_tensor

import matplotlib.pyplot as plt
class DDIMConditionPipeline(DDIMPipeline):
    @torch.no_grad()
    def __call__(self, approx, cond, generator, num_inference_steps=50, eta=0.7):
        batch_size = cond.size(0)
        init_noise = randn_tensor((batch_size, 1, self.unet.config.sample_size, self.unet.config.sample_size), generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        x = torch.cat([init_noise, approx], dim=1)  # [B, 2, H, W] if in_channels=2

        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            pred_noise = self.unet(x, t, cond).sample
            x = self.scheduler.step(
                pred_noise,
                t,
                x,
                eta=eta,
                generator=generator,
                use_clipped_model_output=True
            ).prev_sample
            x = torch.cat([x[:, :1, :, :], approx], dim=1)  # [B, 2, H, W] if in_channels=2

        x = x[:, :1, :, :]
        x = torch.relu(x)
        return ImagePipelineOutput(images=torch_tensor_to_numpy_image(x))


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
            ['particles', 'standard',  lambda p: p[:, None]],
            ['responses', args.responses_scaler, lambda p: p[:, None]],
            [f'approx/{apporx_name}/approx', 'none', lambda p: p[:, None]]
        ]
    )


    enhancer_checkpoints_dir = Path(args.trainings_dir).resolve() / enhancer_name / 'checkpoints'
    os.makedirs(enhancer_checkpoints_dir, exist_ok=True)
    
# Loading Enhancer
    enhancer_model = get_enhancer_model_and_optimizer(args.enhancer_model_id).to(device)
    optimizer, lr_scheduler = get_optimizer(
        enhancer_model,
        epochs=args.epochs,
        batch_size=args.enhancer_train_batch_size,
        n_examples=data_scope['n_examples']['train'],
        lr=args.enhancer_learning_rate)



    if not args.no_resume:
        last_epoch = get_last_epoch(enhancer_checkpoints_dir)
        if last_epoch is not None:
            load_approx_model(enhancer_checkpoints_dir, enhancer_model, last_epoch)
            print(f"Loaded enhancer model from epoch {last_epoch}")
        else:
            last_epoch = 0


    config = {**vars(args)}
    config['model_type'] = str(type(enhancer_model))
    config['model_family'] = 'diffusion-enhancer'
    config['name'] = args.enhancer_name
    config['approx_name'] = apporx_name
    from focal.utils.train import get_num_parameters
    config['model_num_parameters'] = get_num_parameters(enhancer_model)
    
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
    

    diffusion_pipeline = DDIMConditionPipeline(unet=enhancer_model, scheduler=noise_scheduler)

    @torch.no_grad()
    def eval_pipeline(*batch, num_infernece_steps=50):
        cond = batch[0].to(device)
        approx = batch[2].to(device)
        return diffusion_pipeline(approx, cond, generator=generator, num_inference_steps=num_infernece_steps).images

    if not args.ftest:  
        for epoch in trange(last_epoch, args.epochs, desc=f'Epochs {args.name}'):
            enhancer_model.train()
            
            pbar = tqdm(data_scope['dataloaders']['train'], desc="Train loader")
            for batch in pbar:
                optimizer.zero_grad()

                cond, responses, approx = batch
                responses = responses.to(device)
                cond = cond.to(device)
                approx = approx.to(device)

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (responses.shape[0],), generator=generator).to(device).long()
                noise = torch.randn(responses.shape, generator=generator).to(device)
                noisy_images = noise_scheduler.add_noise(responses, noise, timesteps)
                x = torch.cat([noisy_images, approx], dim=1)

                noise_pred = enhancer_model(x, timesteps, cond).sample
                loss = F.mse_loss(noise_pred, noise)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(enhancer_model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                metrics.add({ 'mse_loss': loss.detach().item()}, section='train')

                pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f} Train loader")

            metrics.log(epoch)


            enhancer_model.eval()
            if epoch % args.val_freq == 0:
                with torch.no_grad():
                    num_inference_steps = 50
                    enhancer_model.eval()
                    metrics.current_epoch = epoch
                    eval_step(
                        lambda *batch: eval_pipeline(*batch, num_infernece_steps=num_inference_steps),
                        data_scope['dataloaders']['test'],
                        data_scope,
                        'validation',
                        metrics,
                        tag=f'i{num_inference_steps}'
                    )

                    generated_samples = eval_pipeline(sample_particles, None, sample_approx, num_infernece_steps=num_inference_steps)
                    metrics.plot_responses(
                            [sample_responses, sample_approx_image, generated_samples],
                            epoch,
                            labels=["Original", "Approx", "Generated"],
                            tag=f"i{num_inference_steps}",
                        )

                
            save_approx_model(enhancer_checkpoints_dir, enhancer_model, epoch)
            metrics.log(epoch)            


        # print(f'Testing after {args.epochs} epochs of trainig')
        # with torch.no_grad():
        #     eval_step(eval_pipeline, data_scope['dataloaders']['test'], data_scope, 'test', metrics)
        #     metrics.log(args.epochs)
        #     print(f'Finished training of {enhancer_name}')

        print("Training finished ✔️")

    else:
        from focal.utils.train import get_all_saved_epochs, get_pretrained_from_checkpoint_by_epoch
        from focal.utils.train import find_closest_epoch_file 
        saved_epochs = [find_closest_epoch_file(enhancer_checkpoints_dir, args.ftest)]
        enhancer_model.eval()
        with torch.no_grad():
            for epoch in tqdm(saved_epochs, desc=f'Epochs {enhancer_name}'):
                load_approx_model(enhancer_checkpoints_dir, enhancer_model, epoch=epoch)
                # diffusion_pipeline = get_pretrained_from_checkpoint_by_epoch(enhancer_checkpoints_dir, DDIMConditionPipeline, epoch=epoch).to(device)
                for num_inference_steps in [25, 50]: #100]:
                    def eval_pipeline(*batch):
                        cond = batch[0].to(device)
                        approx = batch[2].to(device)
                        return diffusion_pipeline(approx, cond, generator=generator, num_inference_steps=num_inference_steps).images

                    metrics.current_epoch = epoch
                    eval_step(
                        eval_pipeline,
                        data_scope['dataloaders']['test'],
                        data_scope,
                        'ftest',
                        metrics,
                        tag=f'i{num_inference_steps}',

                    )
                metrics.log(epoch, logAsEpoch=True)
            
