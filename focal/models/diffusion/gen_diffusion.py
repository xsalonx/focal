import os
from argparse import ArgumentParser
import sys

import numpy as np
import torch
from diffusers import DDIMPipeline, DDIMScheduler, ImagePipelineOutput, get_cosine_schedule_with_warmup
from diffusers.utils.torch_utils import randn_tensor

from focal.utils.data import get_samples
from focal.utils.torch import torch_tensor_to_numpy_image

from focal.models.diffusion.diffusion import noise_scheduler

from tqdm import tqdm

from pathlib import Path
import matplotlib.pyplot as plt 

class DDIMConditionPipeline(DDIMPipeline):
    @torch.no_grad()
    def __call__(self, cond, generator, num_inference_steps=50, eta=0.7):
        batch_size = cond.shape[0]
        image_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(image, t, cond).sample
            image = self.scheduler.step(
                model_output, t, image, eta=eta, generator=generator, use_clipped_model_output=True
            ).prev_sample
        image = torch.relu(image)
        return ImagePipelineOutput(images=torch_tensor_to_numpy_image(image))


def gen_samples(
        checkpoint_dir,
        device,
        particles,
        generator_seeds=None,
    ):

    if generator_seeds is None:
        generator_seeds = [0]
    
    pretrained_pipeline = DDIMConditionPipeline.from_pretrained(checkpoint_dir)
    model = pretrained_pipeline.unet
    print(f"loaded from {checkpoint_dir}")

    model.to(device)
    
    with torch.no_grad():
        pipeline = DDIMConditionPipeline(unet=model, scheduler=noise_scheduler)
        generated = []
        for gs in tqdm(generator_seeds):
            generated.append(pipeline(particles.to(device), generator=torch.manual_seed(gs)).images)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--data-dir', required=True, type=str)
    args.add_argument('--checkpoint-dir', required=True, type=str)
    args.add_argument('--epoch', required=False, default=None, type=int)
    args.add_argument('--samples-no', required=False, default=None, type=int)
    args = args.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    gen_samples(
        get_samples(args.data_dir),
        device,
        args.checkpoint_dir,
        args.samples_no,
    )
