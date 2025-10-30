import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data_loader(data_list, batch_size, generator, device, shuffle=True):
    dataset = TensorDataset(*[torch.Tensor(np.array(data)).to(device) for data in data_list])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def torch_tensor_to_numpy_image(x):
    return x.cpu().permute(0, 2, 3, 1).numpy()
