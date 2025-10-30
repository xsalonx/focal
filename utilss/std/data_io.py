import numpy as np
import os
from pathlib import Path
import torch

DEFALT_PARTICLES_NPZ_NAME = 'particles.npz'
DEFALT_RESPONSES_NPZ_NAME = 'responses.npz'
DEFALT_ACCEPTANCE_MASK_NPZ_NAME = 'acceptance-mask.npz'

DEFALT_SAMPLE_PARTICLES_NPZ_NAME = 'sample_' + DEFALT_PARTICLES_NPZ_NAME
DEFALT_SAMPLE_RESPONSES_NPZ_NAME = 'sample_' + DEFALT_RESPONSES_NPZ_NAME


DEFALT_CENTROIDS_NPZ_NAME = 'centroids.npz'
DEFALT_CHARACTERISTICS_NPZ_NAME = 'characteristics.npz'

# def save_focal_data(path, particles, responses, samples_no=128, dont_expand=True, acceptance_mask=None):
#     path = Path(path)
#     os.makedirs(path, exist_ok=True)

#     if not dont_expand:
#         responses = np.expand_dims(responses, axis=3)

#     print(f"particles shape: {particles.shape}")
#     print(f"responses shape: {responses.shape}")

#     np.savez(path / DEFALT_PARTICLES_NPZ_NAME, arr_0=particles)
#     np.savez(path / DEFALT_RESPONSES_NPZ_NAME, arr_0=responses)

#     if acceptance_mask is not None:
#         np.savez(path / DEFALT_ACCEPTANCE_MASK_NPZ_NAME, arr_0=acceptance_mask)

#     np.savez(path / DEFALT_SAMPLE_PARTICLES_NPZ_NAME, arr_0=particles[:samples_no])
#     np.savez(path / DEFALT_SAMPLE_RESPONSES_NPZ_NAME, arr_0=responses[:samples_no])


#     print('saved')

def save_indices(path, indices):
    np.savez(path / 'good_indices.npz', arr_0=indices)

def read_focal_data(path, objects_names=['particles', 'responses', 'centroids', 'characteristics'], astensor=False):
    path = Path(path)
    data_dict = {}
    for name in objects_names:
        object_path = path / f'{name}.npz'
        if object_path.exists():
            data = np.load(object_path)['arr_0']
            if astensor:
                data = torch.Tensor(data)
            data_dict[name] = data
    
    return data_dict

def save_focal_data(path, data_dict):
    path = Path(path)
    for name, data in data_dict.items():
        object_path = path / f'{name}.npz'
        os.makedirs(object_path.parent, exist_ok=True)
        np.savez(object_path, arr_0=data)
        print(f'Saved {name}')

def read_focal_samples(path, astensor=False):
    P = np.load(path / DEFALT_SAMPLE_PARTICLES_NPZ_NAME)['arr_0']
    R = np.load(path / DEFALT_SAMPLE_RESPONSES_NPZ_NAME)['arr_0']

    if astensor:
        P = torch.Tensor(P)
        R = torch.Tensor(R)
    
    return P, R

def save_focal_centroids_and_energy_characteristics(path, centroids, characteristics):
    np.savez(path / DEFALT_CENTROIDS_NPZ_NAME, arr_0=centroids)
    np.savez(path / DEFALT_CHARACTERISTICS_NPZ_NAME, arr_0=characteristics)
    print('saved')

def read_focal_centroids_and_energy_characteristics(path, astensor=False):
    centroids = np.load(path / DEFALT_CENTROIDS_NPZ_NAME)['arr_0']
    characteristics = np.load(path / DEFALT_CHARACTERISTICS_NPZ_NAME)['arr_0']

    if astensor:
        centroids = torch.Tensor(centroids)
        characteristics = torch.Tensor(characteristics)
    
    return centroids, characteristics

def read_indices(path):
    return np.load(path / 'good_indices.npz')['arr_0']
