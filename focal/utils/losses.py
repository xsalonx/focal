from functools import partial

import jax
import jax.numpy as jnp
import optax
from lpips_j.lpips import LPIPS

from focal.models import RESPONSE_SHAPE
from focal.utils.wasserstein import wasserstein_distance
import torch

import jax
import jax.numpy as jnp
from jax import lax

def kl_loss(mean, log_var):
    return -0.5 * (1. + log_var - jnp.square(mean) - jnp.exp(log_var)).sum(axis=1).mean()


def mse_loss(x, y):
    return jnp.square(x - y).reshape(x.shape[0], -1).sum(axis=-1).mean()


def mae_loss(x, y):
    return jnp.abs(x - y).reshape(x.shape[0], -1).sum(axis=-1).mean()

import jax.numpy as jnp
from jax import vmap

# def wasserstein_loss(X, Y):
#     wds = []
#     for x, y in zip(X, Y):
#         wd = wasserstein_distance(x, y)
#         wds.append(wd)

#     return jnp.stack(wds).mean()

def wasserstein_loss(X, Y):
    wasserstein_fn = lambda x, y: jax.lax.stop_gradient(
        jnp.asarray(wasserstein_distance(x, y))
    )
    distances = vmap(wasserstein_fn)(X, Y)  # Vectorize over rows of X and Y

    return jnp.mean(distances)


def xentropy_loss(x, y):
    return optax.sigmoid_binary_cross_entropy(x, y).reshape(x.shape[0], -1).sum(axis=-1).mean()



import numpy as np
import jax
import jax.numpy as jnp
from lpips_j.lpips import LPIPS
from tqdm import tqdm

def perceptual_loss_factory():
    lpips = LPIPS()
    x_sample = jnp.zeros((1, *RESPONSE_SHAPE))
    params = lpips.init(jax.random.PRNGKey(0), x_sample, x_sample)

    def apply(x, y, batch_size=512):
        max_val = max(x.max(), y.max())

        total_loss = 0.0
        n_samples = x.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        x = 2 * (x / max_val) - 1
        y = 2 * (y / max_val) - 1

        for i in tqdm(range(n_batches), desc="Perceptual loss batches"):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            # Prepare chunk
            x_chunk = x[start:end]
            y_chunk = y[start:end]
            # Normalize and move to proper device if necessary

            # Compute LPIPS for chunk (wrap in with torch.no_grad() if you use PyTorch internally)
            loss_chunk = lpips.apply(params, x_chunk, y_chunk).mean()
            total_loss += float(loss_chunk) * (end - start)

        # Return dataset average
        return total_loss / n_samples

    return apply


def sinkhorn_loss(diameter, blur, scaling):
    def eps_schedule(diameter, blur, scaling):
        return jnp.concatenate([
            jnp.asarray([diameter ** 2]),
            jnp.exp(jnp.arange(2 * jnp.log(diameter), 2 * jnp.log(blur), 2 * jnp.log(scaling))),
            jnp.asarray([blur ** 2])
        ])

    def cost(x, y):
        D_xx = (x * x).sum(axis=-1)[:, :, None]
        D_xy = x @ y.transpose(0, 2, 1)
        D_yy = (y * y).sum(axis=-1)[:, None, :]
        return (D_xx - 2 * D_xy + D_yy) / 2

    def softmin(eps, C_xy, h_y):
        return -eps * jax.nn.logsumexp(h_y.reshape(1, 1, -1) - C_xy / eps, axis=2)

    def sinkhorn(x, y, eps_list):
        a, b = jnp.ones(x.shape[0]) / x.shape[0], jnp.ones(y.shape[0]) / y.shape[0]
        x, y, a, b = jax.tree_map(lambda x: x[None, ...], (x, y, a, b))
        a_log, b_log = jnp.log(a), jnp.log(b)

        C_xy = cost(x, jax.lax.stop_gradient(y))
        C_yx = cost(y, jax.lax.stop_gradient(x))
        C_xx = cost(x, jax.lax.stop_gradient(x))
        C_yy = cost(y, jax.lax.stop_gradient(y))

        def sinkhorn_iter():
            eps = eps_list[0]

            g_ab = softmin(eps, C_yx, a_log)
            f_ba = softmin(eps, C_xy, b_log)
            f_aa = softmin(eps, C_xx, a_log)
            g_bb = softmin(eps, C_yy, b_log)

            for i in range(len(eps_list)):
                eps = eps_list[i]

                ft_ba = softmin(eps, C_xy, b_log + g_ab / eps)
                gt_ab = softmin(eps, C_yx, a_log + f_ba / eps)
                ft_aa = softmin(eps, C_xx, a_log + f_aa / eps)
                gt_bb = softmin(eps, C_yy, b_log + g_bb / eps)

                f_ba, g_ab = (f_ba + ft_ba) / 2, (g_ab + gt_ab) / 2
                f_aa, g_bb = (f_aa + ft_aa) / 2, (g_bb + gt_bb) / 2

            return f_aa, g_bb, g_ab, f_ba

        f_aa, g_bb, g_ab, f_ba = jax.lax.stop_gradient(sinkhorn_iter())
        eps = eps_list[-1]

        f_ba, g_ab = (
            softmin(eps, C_xy, jax.lax.stop_gradient(b_log + g_ab / eps)),
            softmin(eps, C_yx, jax.lax.stop_gradient(a_log + f_ba / eps)),
        )

        f_aa = softmin(eps, C_xx, jax.lax.stop_gradient(a_log + f_aa / eps))
        g_bb = softmin(eps, C_yy, jax.lax.stop_gradient(b_log + g_bb / eps))

        return (f_ba - f_aa).mean() + (g_ab - g_bb).mean()

    return partial(sinkhorn, eps_list=eps_schedule(diameter, blur, scaling))



def jax_center_of_mass(responses: jnp.ndarray):
    """
    Parameters:
        responses (jnp.ndarray): of size (samples_no, sample_height, samples_width)

    Return:
        coords (jnp.ndarray[n, 2])
    """
    batch_size, height, width = responses.shape

    y_coords = jnp.arange(height, dtype=jnp.float32)
    x_coords = jnp.arange(width, dtype=jnp.float32)
    y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing="ij")  # shape (H, W)

    # Broadcast grid to (N, H, W)
    y_grid = jnp.expand_dims(y_grid, 0)  # (1, H, W)
    x_grid = jnp.expand_dims(x_grid, 0)

    responses = jnp.clip(responses, 0)
    total_mass = jnp.sum(responses, axis=(1, 2))
    total_mass = jnp.maximum(total_mass, 1e-3)  # Avoid division by zero

    y_com = jnp.sum(responses * y_grid, axis=(1, 2)) / total_mass
    x_com = jnp.sum(responses * x_grid, axis=(1, 2)) / total_mass

    return jnp.stack([x_com, y_com], axis=1)


def L2_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sum(jnp.square(y_true - y_pred), axis=1))

def L1_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sum(jnp.abs(y_true - y_pred), axis=1))

def flat_wasserstein(generated, responses):
    responses, generated = jnp.exp(responses) - 1, jnp.exp(generated) - 1

    return wasserstein_loss(
        responses.reshape(-1, RESPONSE_SHAPE[1] * RESPONSE_SHAPE[0]),
        generated.reshape(-1, RESPONSE_SHAPE[1] * RESPONSE_SHAPE[0]),
    )

def default_eval_fn(generated, responses):
    """
        Take into account that evaluation matrics are calculated upon exp(responses)
    """

    # responses, generated = jnp.exp(responses) - 1, jnp.exp(generated) - 1

    rmse = jnp.sqrt(mse_loss(responses, generated))
    mae = mae_loss(responses, generated)

    wasserstein_1 = wasserstein_loss(
        responses.sum(axis=1).reshape(-1, RESPONSE_SHAPE[1]),
        generated.sum(axis=1).reshape(-1, RESPONSE_SHAPE[1]),
    )

    wasserstein_2 = wasserstein_loss(
        responses.sum(axis=2).reshape(-1, RESPONSE_SHAPE[1]),
        generated.sum(axis=2).reshape(-1, RESPONSE_SHAPE[1]),
    )

    wasserstein_flat = wasserstein_loss(
        responses.reshape(-1, RESPONSE_SHAPE[1] * RESPONSE_SHAPE[0]),
        generated.reshape(-1, RESPONSE_SHAPE[1] * RESPONSE_SHAPE[0]),
    )

    wasserstein_sum = wasserstein_1 + wasserstein_2

    coors = jax_center_of_mass(responses[:, :, :, 0]), jax_center_of_mass(generated[:, :, :, 0])

    return rmse, mae, wasserstein_1, wasserstein_2, wasserstein_sum, wasserstein_flat, L1_loss(*coors), L2_loss(*coors)


from functools import partial

# ---------------- utilities -----------------
def _pad_to_multiple(x, multiple: int):
    pad = (-x.shape[0]) % multiple
    if pad:
        x = jnp.pad(
            x,
            [(0, pad)] + [(0, 0)] * (x.ndim - 1),
            mode="wrap",
        )
    return x, pad

def default_eval_chunked(generated: jnp.ndarray,
                     responses: jnp.ndarray,
                     *,
                     chunk_size: int = 1024):
    """
    Run `default_eval_fn` in constant memory.

    *  Only `chunk_size` samples are live at any point.
    *  Works under `jax.jit`; all loop bounds are static.
    """

    EVAL_METRICS_NUM = 8 # See deafult_eval_fn

    generated, pad_g = _pad_to_multiple(generated, chunk_size)
    responses, pad_r = _pad_to_multiple(responses, chunk_size)
    assert pad_g == pad_r                        # same padding for both

    n_total   = generated.shape[0]               # padded length (static)
    n_chunks  = n_total // chunk_size            # static too
    weights   = jnp.zeros(n_chunks, jnp.float32).at[:-1].set(chunk_size)
    # correct weight for the *true* last chunk
    weights   = weights.at[-1].set(chunk_size - pad_g if pad_g else chunk_size)

    def body(idx, acc):
        start = idx * chunk_size
        g = lax.dynamic_slice_in_dim(generated, start, chunk_size, axis=0)
        r = lax.dynamic_slice_in_dim(responses, start, chunk_size, axis=0)
        acc = acc + jnp.stack(default_eval_fn(g, r)) * weights[idx]
        return acc

    summed = lax.fori_loop(0, n_chunks,
                           body,
                           jnp.zeros((EVAL_METRICS_NUM,), jnp.float32))
    return summed / (n_total - pad_g)            # remove padded rows


default_eval_metrics_names = ('mse', 'mae', 'wasserstein_1', 'wasserstein_2', 'wasserstein_sum', 'wasserstein_flat', 'positional_loss_l1', 'positional_loss_l2')
