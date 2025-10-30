
import os
import torch

from pathlib import Path 

def _unwrap(model):
    # handles DataParallel / DistributedDataParallel without being a diva
    return getattr(model, "module", model)

def save_gan(checkpoints_dir, G, D, epoch, G_opt=None, D_opt=None, **extra):
    """
    Save generator & discriminator (and optional optimizers + metadata).
    Use .pt/.pth extension by convention.
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoint_dir = checkpoints_dir / f'epoch_{epoch}'
    checkpoint_path = checkpoint_dir / 'state'

    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt = {
        "G_state": _unwrap(G).state_dict(),
        "D_state": _unwrap(D).state_dict(),
        "epoch": epoch,
        "extra": extra or {}
    }
    if G_opt is not None:
        ckpt["G_opt_state"] = G_opt.state_dict()
    if D_opt is not None:
        ckpt["D_opt_state"] = D_opt.state_dict()

    torch.save(ckpt, checkpoint_path)


def load_gan(checkpoints_dir, G, D, epoch, G_opt=None, D_opt=None, map_location="auto", strict=True):
    """
    Load weights into G and D, optionally restore optimizers.
    Returns {'epoch': ..., 'extra': {...}}.
    """

    checkpoints_dir = Path(checkpoints_dir)
    checkpoint_dir = checkpoints_dir / f'epoch_{epoch}'
    checkpoint_path = checkpoint_dir / 'state'

    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=map_location)
    _unwrap(G).load_state_dict(ckpt["G_state"], strict=strict)
    if D:
        _unwrap(D).load_state_dict(ckpt["D_state"], strict=strict)

    # return ckpt

    if G_opt is not None and "G_opt_state" in ckpt:
        G_opt.load_state_dict(ckpt["G_opt_state"])
    if D_opt is not None and "D_opt_state" in ckpt:
        D_opt.load_state_dict(ckpt["D_opt_state"])

    return { 'gen': G, 'disc': D, 'G_opt': G_opt, 'D_opt': D_opt }
