import time
import os

import jax
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange, tqdm

from focal.utils.torch import torch_tensor_to_numpy_image

from focal.utils.losses import default_eval_metrics_names, default_eval_chunked, perceptual_loss_factory

perceptual_loss = perceptual_loss_factory()

EVAL_CHUNK_SIZES = [128, 32, 8] if 'MIN_EVAL_CHUNKS' in os.environ else [512, 128, 32, 8]

LAST_VALID_CHUNK_SIZE = { 'DEFAULT_LOSS': None, 'PERC_LOSS': None }

import ot

def calc_OT(P, R, R_gen):
    P = P.reshape((-1, 1))
    R = R.reshape((-1, 1))
    R_gen = R_gen.reshape((-1, 1))

    X = np.concatenate([P, R], axis=1)
    Y = np.concatenate([P, R_gen], axis=1)

    a = np.ones((X.shape[0], )) / X.shape[0]
    b = np.ones((Y.shape[0], )) / Y.shape[0]
    M = ot.dist(X, Y)
    reg = 1

    return ot.sinkhorn2(a, b, M, reg)    

from pprint import pprint

import numpy as np

import numpy as np
import os
import re
from pathlib import Path


ENV_OMIT_PERCETUAL_LOSS = os.environ['OMIT_PERCETUAL_LOSS'] if 'OMIT_PERCETUAL_LOSS' in os.environ else False
ENV_OMIT_PERCETUAL_LOSS = ENV_OMIT_PERCETUAL_LOSS in ['true', "True", "t", "T"]

ENV_OMIT_TIME_MEAS = os.environ['OMIT_TIME_MEAS'] if 'OMIT_TIME_MEAS' in os.environ else False
ENV_OMIT_TIME_MEAS = ENV_OMIT_TIME_MEAS in ['true', "True", "t", "T"]

SLURM_CLUSTER_NAME = os.environ['SLURM_CLUSTER_NAME'] if 'SLURM_CLUSTER_NAME' in os.environ else ''


ENV_NO_EVAL_CACHE = os.environ['ENV_NO_EVAL_CACHE'] if 'ENV_NO_EVAL_CACHE' in os.environ else False
ENV_NO_EVAL_CACHE = ENV_NO_EVAL_CACHE in ['true', "True", "t", "T"]

ENV_OVERRIDE_EVAL_CACHE = os.environ['ENV_OVERRIDE_EVAL_CACHE'] if 'ENV_OVERRIDE_EVAL_CACHE' in os.environ else False
ENV_OVERRIDE_EVAL_CACHE = ENV_OVERRIDE_EVAL_CACHE in ['true', "True", "t", "T"]


def save_arrays_keep_latest(path, conds, original, generated, generation_times, generated_numbers, cluster_name='', keep=3):
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    # Save arrays, including generation_times
    arr_conds = np.array(conds, dtype=object)
    arr_original = np.array(original, dtype=np.float64)
    arr_generated = np.array(generated, dtype=np.float64)
    arr_gen_times = np.array(generation_times, dtype=np.float64)
    arr_generated_numbers = np.array(generated_numbers, np.int32)

    np.savez(path, conds=arr_conds, original=arr_original,
             generated=arr_generated, generation_times=arr_gen_times, cluster_name=cluster_name, generated_numbers=arr_generated_numbers)
    
    
    file_pattern = re.compile(r"epoch_(\d+)\.npz$")
    files_and_epochs = []
    for fname in os.listdir(dir_path):
        match = file_pattern.search(fname)
        if match:
            epoch = int(match.group(1))
            files_and_epochs.append((epoch, os.path.join(dir_path, fname)))
    files_and_epochs.sort(reverse=True)
    for _, fpath in files_and_epochs[keep:]:
        try:
            os.remove(fpath)
        except Exception as e:
            print(f"Could not remove file {fpath}: {e}")


def load_eval_cache(path):
    data = np.load(path, allow_pickle=True)
    conds = data['conds']
    original = np.array(data['original'], dtype=np.float64)
    generated = np.array(data['generated'], dtype=np.float64)
    generation_times = data['generation_times']
    cluster_name = data['cluster_name']
    generated_numbers = data['generated_numbers']
    return conds, original, generated, generation_times, cluster_name, generated_numbers


def normalize_minmax(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))




def sliced_w2_joint(P, R, R_gen, n_proj: int = 1024, seed: int = 0, standardize = None):
    """
    Fast approximation of 2D W2 between empirical measures of X=[P,R] and Y=[P,R_gen].
    Complexity ~ O(n_proj * N log N), no NxN matrix.
    """
    P = np.asarray(P).reshape(-1, 1).astype(np.float32)
    R = np.asarray(R).reshape(-1, 1).astype(np.float32)
    Rg = np.asarray(R_gen).reshape(-1, 1).astype(np.float32)

    X = np.concatenate([P, R], axis=1)    # [N,2]
    Y = np.concatenate([P, Rg], axis=1)   # [N,2]

    if standardize == 'standard':
        Z = np.vstack([X, Y])
        mu = Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, keepdims=True); sd[sd < 1e-8] = 1.0
        X = (X - mu) / sd
        Y = (Y - mu) / sd

    elif standardize == 'Pmax':
        X /= P.max()
        Y /= P.max()

    elif standardize == 'max':
        X /= X.max(axis=0)
        Y /= Y.max(axis=0)

    rng = np.random.RandomState(seed)
    # Random directions on S^1
    theta = rng.uniform(0.0, 2.0*np.pi, size=(n_proj,))
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)  # [K,2]

    w2s = []
    for v in dirs:
        x1d = X @ v
        y1d = Y @ v
        x1d.sort(); y1d.sort()
        diff = x1d - y1d
        w2s.append(float(np.mean(diff*diff)))  # 1D W2^2
    return float(np.mean(w2s))  # sliced W2^2


def image_prepare(v, deg=1):
    for _ in range(deg):
        v = np.log1p(v)
    return v

max_images = 7

prepare_images = lambda vs, deg: [image_prepare(v[:max_images], deg=deg) for v in vs]

def plot_samples(data_scope, pipeline, metrics, tag, section):
    samples_batch = [torch.Tensor(d) for d in data_scope['samples']]
    samples = data_scope['samples']
    names = data_scope['data_names']

    scalers = data_scope['scalers']

    responses_index = names.index('responses')
    approx_index = names.index('approx') if 'approx' in names else None

    responses = scalers[responses_index].inverse_transform(samples[responses_index]).squeeze()
    approx = scalers[approx_index].inverse_transform(samples[approx_index]).squeeze() if approx_index is not None else None

    generated_reponses = pipeline(*samples_batch)
    generated_reponses = scalers[responses_index].inverse_transform(generated_reponses).squeeze()

 
    for deg in [1, 3]:
        if approx is None:
            metrics.plot_responses(
                    prepare_images([responses, generated_reponses], deg=deg),
                    labels=["Original", "Generated"],
                    tag=tag + f"_{deg}log1p", section=section,
                )
        else:
            metrics.plot_responses(
                    prepare_images([responses, approx, generated_reponses], deg=deg),
                    labels=["Original", "Approx", "Generated"],
                    tag=tag + f"_{deg}log1p", section=section,
                )


import numpy as np

def remove_x_largest(arrays, R_gen, outliers_num):
    # Find indices of X largest values in R_gen
    indices = np.argpartition(R_gen, -outliers_num)[-outliers_num:]
    mask = np.ones(R_gen.shape[0], dtype=bool)
    mask[indices] = False
    # Filter all input arrays using the mask
    return [arr[mask] for arr in arrays]



@torch.no_grad()
def eval_step(
        pipeline,
        dataloader,
        data_scope,
        section,
        metrics,
        tag='',
        n_rep=1,
        eval_fn=jax.jit(default_eval_chunked, static_argnames=("chunk_size",)),
        eval_metrics_names=default_eval_metrics_names,
        no_energy_dist_plot=False,
        calc_perceptual_loss = True,
        silent=False,
    ):

    if section == 'ftest' or section == 'test':
        section = 'ftest2'


    if metrics.config is not None and 'data_dir' in metrics.config and metrics.current_epoch is not None and 'validation' not in section and not ENV_NO_EVAL_CACHE:
        cache_path = Path(metrics.config['data_dir']) / 'eval_cache_v2' / (metrics.run_name + tag) / f"epoch_{metrics.current_epoch}.npz"
    else:
        cache_path = None


    conds, original, generated, generation_times, prev_cluster_name, generated_numbers = [], [], [], [], None, []
    if cache_path is not None and os.path.exists(cache_path) and not ENV_OVERRIDE_EVAL_CACHE:
        conds, original, generated, generation_times, prev_cluster_name, generated_numbers = load_eval_cache(cache_path)
    else:
        for batch in tqdm(dataloader, desc=f'{section} loader'):
            for rep_no in range(n_rep):
                print(f"Doing generation repetion ({section}) ({rep_no + 1}/{n_rep})")
                cond, responses, *_ = batch
                if cond.size(0) == 1:
                    continue

                batch = [d.to('cuda') for d in batch] # not to count it as time
                time_generation_pre = time.time()
                output = pipeline(*batch)
                time_generation_post = time.time()

                if isinstance(output, torch.Tensor):
                    output = output.cpu().numpy()
                    
                generation_times.append(time_generation_post - time_generation_pre)
                generated_numbers.append(output.shape[0])
                generated.append(output)
                original.append(responses)
                conds.append(cond.squeeze(1).cpu().numpy())


        generated, original = np.concatenate(generated, dtype=np.float32), np.concatenate([torch_tensor_to_numpy_image(xs) for xs in original], dtype=np.float32)
    
        if cache_path is not None:
            save_arrays_keep_latest(cache_path, conds, original, generated, generation_times, cluster_name=SLURM_CLUSTER_NAME, generated_numbers=generated_numbers, keep=3)


    if not ENV_OMIT_TIME_MEAS:
        if not prev_cluster_name or prev_cluster_name == SLURM_CLUSTER_NAME:

            generation_times = np.array(generation_times)
            generated_numbers = np.array(generated_numbers)
            weighted_generation_times_per_sample = generation_times / generated_numbers

            metrics.add({
                'total_loader_time_generation_mean': weighted_generation_times_per_sample.mean() * generated_numbers.sum(),
                'total_loader_time_generation_min': weighted_generation_times_per_sample.min() * generated_numbers.sum(),
                'total_loader_time_generation_max': weighted_generation_times_per_sample.max() * generated_numbers.sum(),

                'time_generation_per_sample_mean': weighted_generation_times_per_sample.mean(),
                'time_generation_per_sample_var': weighted_generation_times_per_sample.var(),
                'time_generation_per_sample_min': weighted_generation_times_per_sample.min(),
                'time_generation_per_sample_max': weighted_generation_times_per_sample.max(),

                'generated_count': generated_numbers.sum(),
                }, suffix=f"{tag}_{SLURM_CLUSTER_NAME}", section=section)
        else:
            print("Cached evaluations were done on different cluster")
    else:
        print("Time measurments omitted")


    if 'test' in section:
        plot_samples(data_scope, pipeline, metrics, tag, section)



    names = data_scope['data_names']
    scalers = data_scope['scalers']
    responses_index = names.index('responses')
    particles_index = names.index('particles')


    # In original domain
    responses_scaler = scalers[responses_index]
    generated, original = responses_scaler.inverse_transform(generated), responses_scaler.inverse_transform(original)
    Pe = scalers[particles_index].inverse_transform(np.concatenate(conds))[:, 0]
    R_gen = generated.sum(axis=(1, 2))
    
    generated_raw, original_raw, Pe = remove_x_largest([generated, original, Pe], R_gen, 1)
    
    
    R_org = original_raw.squeeze().sum(axis=(1, 2))
    R_gen = generated_raw.squeeze().sum(axis=(1, 2))

    try:
        if not no_energy_dist_plot:
            metrics.plot_energy_ratio_distributions(
                [Pe, R_org, R_gen],
                section=section,
                suffix=tag,
            )
    except Exception as e:
        pprint(e)

    metrics.add({
        'total_energy_responses_wasserstein_1d': ot.wasserstein_1d(R_org, R_gen),
        'total_energy_responses_wasserstein_1d_minmax': ot.wasserstein_1d(normalize_minmax(R_org), normalize_minmax(R_gen)),
        'total_energy_responses_wasserstein_1d_max': ot.wasserstein_1d(R_org / R_org.max(), R_gen / R_gen.max()),
        'total_energy_responses_wasserstein_1d_Pmax': ot.wasserstein_1d(R_org / Pe.max(), R_gen / Pe.max()),

        'energy_ratio_distirubiton_sinkhorn': calc_OT(Pe, R_org, R_gen),
        'energy_ratio_distirubiton_sinkhorn_max': calc_OT(Pe / Pe.max(), R_org / R_org.max(), R_gen / R_gen.max()),
        'energy_ratio_distirubiton_sinkhorn_Pmax': calc_OT(Pe / Pe.max(), R_org / Pe.max(), R_gen / Pe.max()),

        'energy_raio_distirubiton_sliced_w2_joint': sliced_w2_joint(Pe, R_org, R_gen),
        'energy_raio_distirubiton_sliced_w2_joint_std': sliced_w2_joint(Pe, R_org, R_gen, standardize='standard'),
        'energy_raio_distirubiton_sliced_w2_joint_max': sliced_w2_joint(Pe, R_org, R_gen, standardize='max'),
        'energy_raio_distirubiton_sliced_w2_joint_Pmax': sliced_w2_joint(Pe, R_org, R_gen, standardize='Pmax'),
    }, suffix=tag, section=section)



    raw_tag = tag
    for domain_squeeze_degree in [0, 1, 3]:
        print(f"---------Domain squeeze {domain_squeeze_degree}-------------")
        generated = image_prepare(generated_raw, deg=domain_squeeze_degree)
        original = image_prepare(original_raw, deg=domain_squeeze_degree)
        tag = raw_tag + f'_squeeze-{domain_squeeze_degree}log1p'

        global LAST_VALID_CHUNK_SIZE
        if not silent:
            print(f'[{section}] Calculating {section} metrics on {generated.shape[0]} samples of total size {generated.size} and {generated.nbytes} bytes')
        for chunk_size in EVAL_CHUNK_SIZES if LAST_VALID_CHUNK_SIZE['DEFAULT_LOSS'] is None else [LAST_VALID_CHUNK_SIZE['DEFAULT_LOSS']]:
            try:
                if not silent:
                    print(f"Trying eval_fn with chunks size {chunk_size}")
                eval_results = eval_fn(generated, original, chunk_size=chunk_size)
                metrics.add(dict(zip(eval_metrics_names, eval_results)), suffix=tag, section=section)
                if not silent:
                    print(f'Added default eval {section} metrics for squeeze {domain_squeeze_degree}')

                LAST_VALID_CHUNK_SIZE['DEFAULT_LOSS'] = chunk_size
                break
            except Exception as e:
                pass

        if not ENV_OMIT_PERCETUAL_LOSS:
            if calc_perceptual_loss: 
                for chunk_size in EVAL_CHUNK_SIZES if LAST_VALID_CHUNK_SIZE['PERC_LOSS'] is None else [LAST_VALID_CHUNK_SIZE['PERC_LOSS']]:
                    try:
                        if not silent:
                            print(f"Trying perceptual loss with chunks size {chunk_size}")
                        with torch.no_grad():
                            metrics.add({
                                'perceptual_loss': perceptual_loss(generated.astype(np.float32), original.astype(np.float32), batch_size=chunk_size),
                            }, suffix=tag, section=section)
                            if not silent:
                                print(f'Added perceptual loss {section} metrics for squeeze {domain_squeeze_degree}')

                        LAST_VALID_CHUNK_SIZE['PERC_LOSS'] = chunk_size
                        break
                    except Exception as e:
                        pass
            else:
                print('Perceptual loss disabled')
        else:
            print('Perceptual loss omitted')

