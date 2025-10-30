import os
from pathlib import Path

import torch

import jax
import jax.numpy as jnp
from tqdm import trange

from focal.models import RESPONSE_SHAPE
from focal.utils.data import batches
from focal.utils.losses import mae_loss, mse_loss, wasserstein_loss
from focal.utils.metrics import Metrics
from focal.utils.nn import save_model, forward

from scipy.stats import wasserstein_distance
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import lax

import re
from pathlib import Path
import torch, tempfile, os, re

from pathlib import Path
import tempfile, os, re, torch
from typing import Optional


import os
import subprocess

def get_slurm_seconds_left_secs():
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None  # not running in SLURM
    # Sformatowane w 'dd-hh:mm:ss', 'hh:mm:ss', 'mm:ss'
    time_left = subprocess.check_output(
        ["squeue", "-h", "-j", job_id, "-O", "TimeLeft"]
    ).decode().strip()
    # Parsowanie czasu
    parts = time_left.split("-")
    if len(parts) == 2:  # dd-hh:mm:ss
        days = int(parts[0])
        hms = parts[1]
    else:
        days = 0
        hms = parts[0]
    hms_split = hms.split(":")
    h, m, s = 0, 0, 0
    if len(hms_split) == 3:
        h, m, s = map(int, hms_split)
    elif len(hms_split) == 2:
        m, s = map(int, hms_split)
    seconds = days*86400 + h*3600 + m*60 + s
    return seconds





def _ckpt_path(checkpoints_dir, epoch, fmt="epoch_{epoch:04d}.pth"):
    d = Path(checkpoints_dir); d.mkdir(parents=True, exist_ok=True)
    return d / fmt.format(epoch=int(epoch))

def _to_device_dtype(obj, device="cpu", dtype: Optional[torch.dtype]=None):
    """Rekurencyjnie przenieś tensory na device i (opcjonalnie) rzutuj typ (tylko floating)."""
    if torch.is_tensor(obj):
        t = obj.detach().to(device)
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        return t
    if isinstance(obj, dict):
        return {k: _to_device_dtype(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device_dtype(v, device, dtype) for v in obj)
    return obj

def _purge_old(checkpoints_dir, pattern="epoch_*.pth", max_to_keep: int = 3):
    files = sorted(Path(checkpoints_dir).glob(pattern))
    if len(files) > max_to_keep:
        to_delete = files[: len(files) - max_to_keep]
        for f in to_delete:
            try: f.unlink()
            except FileNotFoundError: pass

def save_ckpt(checkpoints_dir,
              model,
              optimizer=None,
              scheduler=None,
              epoch: int = 0,
              extra=None,
              *,
              weights_only: bool = False,
              save_dtype: Optional[torch.dtype] = None,  # np. torch.float16, torch.bfloat16
              cpu_offload: bool = True,
              max_to_keep: Optional[int] = 5,
              filename_fmt: str = "epoch_{epoch:04d}.pth"):
    """Elastyczny zapis: kompresja rozmiaru przez rzutowanie dtype i pomijanie opt/sched."""
    path = _ckpt_path(checkpoints_dir, epoch, fmt=filename_fmt)

    # model state
    model_state = model.state_dict()
    device = "cpu" if cpu_offload else next(model.parameters()).device
    model_state = _to_device_dtype(model_state, device=device, dtype=save_dtype)

    payload = {"epoch": int(epoch), "model": model_state, "extra": extra, "torch_version": torch.__version__}

    if not weights_only:
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            # rzutuj bufory optymalizatora (bardzo redukuje rozmiar) – opcjonalne
            opt_state = _to_device_dtype(opt_state, device=device, dtype=save_dtype)
        else:
            opt_state = None
        payload["optimizer"] = opt_state
        payload["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    else:
        payload["optimizer"] = None
        payload["scheduler"] = None

    # atomic save
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try: tmp_path.unlink()
            except: pass

    # Retencja
    if max_to_keep is not None and max_to_keep > 0:
        _purge_old(path.parent, pattern=filename_fmt.replace("{epoch:04d}", "*"), max_to_keep=max_to_keep)
    return path

def load_ckpt(checkpoints_dir,
              model,
              optimizer=None,
              scheduler=None,
              epoch: Optional[int] = None,
              map_location="cpu",
              filename_fmt: str = "epoch_{epoch:04d}.pth",
              strict: bool = True):
    """Wczytaj; jeśli epoch=None, weź najnowszy."""
    d = Path(checkpoints_dir)
    if epoch is None:
        files = sorted(d.glob(filename_fmt.replace("{epoch:04d}", "*")))
        if not files:
            raise FileNotFoundError(f"No checkpoints in {d}")
        path = files[-1]
    else:
        path = _ckpt_path(checkpoints_dir, epoch, fmt=filename_fmt)

    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"], strict=strict)

    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state.get("epoch", epoch or -1), state.get("extra")


def find_closest_epoch_file(dir_path, ep):
    # Compile regex to match files like epoch_<number>
    pattern = re.compile(r'^epoch_(\d+)$')
    closest_file = None
    closest_val = None

    # Loop over all files in the directory
    for fname in os.listdir(dir_path):
        match = pattern.match(fname)
        if match:
            val = int(match.group(1))
            if closest_val is None or abs(val - ep) < abs(closest_val - ep):
                closest_val = val
                closest_file = fname

    return int(closest_file.split('_')[1])

def default_generate_fn(model):
    def generate_fn(params, state, key, *x):
        return forward(model, params, state, key, x[1], method='gen')[0]

    return generate_fn


def get_all_saved_epochs(checkpoints_dir, last_best=0.3, step=2):
    if not os.path.exists(checkpoints_dir):
        return None
    
    checkpoint_dir_names = os.listdir(checkpoints_dir)
    if checkpoint_dir_names:
        checkpoints_epochs = [int(n.split('_')[1]) for n in checkpoint_dir_names]
        checkpoints_epochs.sort()
        epochs_num = len(checkpoints_epochs)
        return checkpoints_epochs[-int(last_best * epochs_num):epochs_num:step]
    
    return None


def get_last_epoch(checkpoints_dir):
    epochs = get_all_saved_epochs(checkpoints_dir, last_best=1, step=1)
    if epochs:
        return max(*(epochs * 2))


def get_pretrained_from_checkpoint(checkpoints_dir, clazz):
    last_epoch_id = get_last_epoch(checkpoints_dir)
    if last_epoch_id > 0:
        last_epoch_storage_object = Path(checkpoints_dir) / f'epoch_{last_epoch_id}'
        if hasattr(clazz, 'from_pretrained'):
            model = clazz.from_pretrained(last_epoch_storage_object)
        else:
            ckpt = torch.load(last_epoch_storage_object, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt)
            model = clazz()
            model.load_state_dict(state)
        print(f"loaded from {last_epoch_storage_object}")
    else:
        model = None

    return model

def get_pretrained_from_checkpoint_by_epoch(checkpoints_dir, clazz, epoch):

    last_epoch_storage_object = Path(checkpoints_dir) / f'epoch_{epoch}'
    if hasattr(clazz, 'from_pretrained'):
        model = clazz.from_pretrained(str(last_epoch_storage_object))
    else:
        ckpt = torch.load(str(last_epoch_storage_object), map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model = clazz()
        model.load_state_dict(state)
    print(f"loaded from {last_epoch_storage_object}")

    return model

def get_num_parameters(model):
    # Use model.num_parameters() if present
    if hasattr(model, 'num_parameters') and callable(getattr(model, 'num_parameters')):
        return model.num_parameters()
    # Otherwise, sum up all parameters
    return sum(p.numel() for p in model.parameters())



def train_loop(
    config,
    init_step,
    data_scope,
    train_step,
    eval_step,
    sample_generation_step,
    save_model_step,
    test_step,
    epochs,
):
    last_epoch = init_step()
    for epoch in trange(last_epoch, epochs, desc='Epochs'):
        for batch in data_scope['dataloaders']['train']:
            # params, opt_state, (state, *losses) = train_fn(batch)
            train_step(batch)
            # metrics.add(dict(zip(train_metrics, losses)), 'train')

        metrics.current_epoch = epoch
        eval_step(data_scope['dataloaders']['train'])
        sample_generation_step()
        save_model_step()

    test_step(data_scope['dataloaders']['train'])


# def train_loop(
#         name, train_fn, eval_fn, generate_fn, train_dataset, val_dataset, test_dataset,
#         train_metrics, eval_metrics, params, state, opt_state, key, epochs=100, batch_size=256, n_rep=5, load_pdgid=False
# ):
#     if eval_fn is None:
#         eval_fn = default_eval_fn
#         eval_metrics = ('rmse', 'mae', 'wasserstein')

#     metrics = Metrics(job_type='train', name=name)
#     os.makedirs(f'checkpoints/{name}', exist_ok=True)

#     train_key, val_key, test_key, shuffle_key, plot_key = jax.random.split(key, 5)
#     samples = get_samples(load_pdgid=load_pdgid)

#     eval_fn = jax.jit(eval_fn)

#     for epoch in trange(epochs, desc='Epochs'):
#         shuffle_key, shuffle_train_subkey, shuffle_val_subkey = jax.random.split(shuffle_key, 3)

#         for batch in batches(*train_dataset, batch_size=batch_size, shuffle_key=shuffle_train_subkey):
#             train_key, subkey = jax.random.split(train_key)
#             params, opt_state, (state, *losses) = train_fn(params, (state, subkey, *batch), opt_state)
#             metrics.add(dict(zip(train_metrics, losses)), 'train')

#         metrics.log(epoch)
#         generated, original = [], []

#         for batch in batches(*val_dataset, batch_size=batch_size, shuffle_key=shuffle_val_subkey):
#             for _ in range(n_rep):
#                 val_key, subkey = jax.random.split(val_key)
#                 generated.append(generate_fn(params, state, subkey, *batch))
#                 original.append(batch)

#         generated, original = jnp.concatenate(generated), (jnp.concatenate(xs) for xs in zip(*original))
#         metrics.add(dict(zip(eval_metrics, eval_fn(generated, *original))), 'val')
#         metrics.log(epoch)

#         if generated.shape[1:] == RESPONSE_SHAPE:
#             plot_key, subkey = jax.random.split(plot_key)
#             metrics.plot_responses(
#                 [samples[0], generate_fn(params, state, subkey, *samples)],
#                 epoch,
#                 labels=['Original', 'Generated'],
#             )

#         save_model(params, state, f'checkpoints/{name}/epoch_{epoch + 1}.pkl.lz4')

#     generated, original = [], []

#     for batch in batches(*test_dataset, batch_size=batch_size):
#         for _ in range(n_rep):
#             test_key, subkey = jax.random.split(test_key)
#             generated.append(generate_fn(params, state, subkey, *batch))
#             original.append(batch)

#     generated, original = jnp.concatenate(generated), (jnp.concatenate(xs) for xs in zip(*original))
#     metrics.add(dict(zip(eval_metrics, eval_fn(generated, *original))), 'test')
#     metrics.log(epochs)

