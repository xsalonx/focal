
import torch
from pathlib import Path

def save_approx_model(checkpoints_dir, model, epoch):
    checkpoints_dir = Path(checkpoints_dir)
    torch.save({'model_state_dict': model.state_dict()}, checkpoints_dir / f'epoch_{epoch + 1}')

def load_approx_model(checkpoints_dir, model, epoch):
    checkpoints_dir = Path(checkpoints_dir)
    model.load_state_dict(torch.load(checkpoints_dir / f'epoch_{epoch}').get('model_state_dict'))
