import torch
from diffusers import DDIMPipeline, ImagePipelineOutput

from focal.utils.torch import torch_tensor_to_numpy_image
from diffusers.utils.torch_utils import randn_tensor




class DDIMConditionPipeline(DDIMPipeline):
    @torch.no_grad()
    def __call__(self, cond, generator=None, num_inference_steps=50, eta=0.7):
        batch_size = cond.shape[0]
        image_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(image, t, cond).sample
            image = self.scheduler.step(
                model_output, t, image, eta=eta, generator=generator, use_clipped_model_output=True
            ).prev_sample
        image = torch.relu(image) # WELL
        return ImagePipelineOutput(images=torch_tensor_to_numpy_image(image))
    


def remove_noise( # based on DDIMScheduler.add_noise method
    scheduler,
    noisy_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.IntTensor,
) -> torch.Tensor:

    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_samples.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=noisy_samples.dtype)
    timesteps = timesteps.to(noisy_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    original_samples = (noisy_samples - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
    return original_samples

def add_dump_metrics(metric_values, comp_coords, coords):

    metric_values['physical_comp_coords_min_x'] = comp_coords[:, 0].min()
    metric_values['physical_comp_coords_max_x'] = comp_coords[:, 0].max()
    metric_values['physical_comp_coords_min_y'] = comp_coords[:, 1].min()
    metric_values['physical_comp_coords_max_y'] = comp_coords[:, 1].max()

    metric_values['physical_coords_min_x'] = coords[:, 0].min()
    metric_values['physical_coords_max_x'] = coords[:, 0].max()
    metric_values['physical_coords_min_y'] = coords[:, 1].min()
    metric_values['physical_coords_max_y'] = coords[:, 1].max()

    # metric_values['physical_coords_mask_valid_values'] = mask.sum()
