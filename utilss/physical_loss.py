import torch

def compute_expected_hit_position(particles: torch.Tensor, z=7.17):
    eta = particles[..., 1]
    phi = particles[..., 2]
    common = z * torch.tan(2 * torch.atan(torch.exp(eta)))
    x = common * torch.sin(phi) ## cos and sin are swaped
    y = common * torch.cos(phi)
    return torch.stack([x, y], axis=1)

def position_to_coords(pos: torch.Tensor):
    cell_size = 6.55 / 7 * 0.01
    x, y = pos[..., 0], pos[..., 1]
    x_coord = x / cell_size
    y_coord = y / cell_size
    x_coord = x_coord * (-1) + 52
    y_coord = y_coord * (-1) + 52
    return torch.stack([x_coord, y_coord], axis=1).type(torch.int32)

def compute_expected_hit_coors(particles: torch.Tensor, z=7.25): # previous z=7.17
    xy = compute_expected_hit_position(particles, z=z)
    computed_coords = position_to_coords(xy)
    return computed_coords



def center_of_mass(responses: torch.Tensor):
    """
    Parameters:
        responses (numpy.array, torch.tensor): of size (samples_no, sample_height, samples_width)

    Return:
        coords (Tensor[n, 2])
    """
    _batch_size, height, width = responses.shape
    
    y_coords = torch.arange(0, height, dtype=torch.float32).to(responses.device)
    x_coords = torch.arange(0, width, dtype=torch.float32).to(responses.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Compute total mass for each image
    total_mass = torch.sum(responses, axis=[1, 2])

    # Avoid division by zero
    total_mass = torch.maximum(total_mass, torch.Tensor([1e-3]).to(total_mass.device))

    # Compute weighted sums of coordinates
    y_com = torch.sum(responses * y_grid, axis=[1, 2]) / total_mass
    x_com = torch.sum(responses * x_grid, axis=[1, 2]) / total_mass

    # Return the centers of mass as (N, 2)
    return torch.stack([x_com, y_com], axis=1).squeeze()


def L2_loss(y_true, y_pred):
    return torch.mean(torch.sum(torch.square(y_true - y_pred), axis=1))

def L1_loss(y_true, y_pred):
    return torch.mean(torch.sum(torch.abs(y_true - y_pred), axis=1))


def calc_positional_loss(particles: torch.Tensor, responses: torch.Tensor, loss_type: str):
    """
    This method is dedicated for single-shower images

    Parameters:
        particles,
        responses,
        loss_type (str): L1 or L2

    Return:
        loss                     (torch.loss),
        expected_coords          (Tensor[n, 2]),
        coords_of_center_of_mass (Tensor[n, 2]
    """
    expected_coords = compute_expected_hit_coors(particles).to(responses.device)
    coords_of_center_of_mass = center_of_mass(responses).to(responses.device)
    

    # x_in_range = coords[:, 0] >= 0
    # x_in_range = torch.logical_and(coords[:, 0] < 105, x_in_range)

    # y_in_range = coords[:, 1] >= 0
    # y_in_range = torch.logical_and(coords[:, 1] < 105, y_in_range)

    # mask = torch.logical_and(x_in_range, y_in_range)
    # if ignore_rubbish:
    #     coords = coords[mask]
    #     computed_coords = computed_coords[mask]

    loss = L1_loss(expected_coords, coords_of_center_of_mass) if loss_type == "L1" else L2_loss(expected_coords, coords_of_center_of_mass)

    return loss, expected_coords, coords_of_center_of_mass

