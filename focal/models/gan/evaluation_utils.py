import numpy as np
import math


def find_radius_circle_bisection(response, diam_to_indices, fraction=0.9):
    response = response.reshape((105, 105))
    energy_target = fraction * response.sum()
    img_size = response.shape[-1]
    max_diam = int(2 * img_size * math.sqrt(2))
    if energy_target == 0:
        return max_diam
    com = np.argwhere(response == response.max())[0]
    center = response.shape
    low_diam, diam = 1, 1
    high_diam = int(fraction * img_size)  # a simple guess to start the bisection algorithm
    checked_diams = [1]
    #current_sum = response[com[0], com[1]].sum()
    while True:
        low_diam, new_diam, high_diam, current_sum = compute_diams(response, diam_to_indices, energy_target, low_diam, high_diam, com, center)
        if new_diam in checked_diams or abs(current_sum - energy_target) / energy_target < 0.02:
            diam = new_diam
            break
        checked_diams.append(new_diam)
    while current_sum >= energy_target and diam > 1:
        #j += 1
        diam -= max(math.ceil(math.sqrt(current_sum / energy_target)), 1)
        diam = max(1, diam)
        mask_indices = move_circle_to_coords(diam_to_indices[diam], center, com)
        current_sum = response[mask_indices[0], mask_indices[1]].sum()
    while current_sum < energy_target and diam < max_diam:
        diam += max(math.ceil(math.sqrt(energy_target / current_sum)), 1)
        diam = min(diam, max_diam)
        mask_indices = move_circle_to_coords(diam_to_indices[diam], center, com)
        current_sum = response[mask_indices[0], mask_indices[1]].sum()
    return diam / 2


def compute_diams(response, diam_to_indices, energy_target, low_diam, high_diam, com, center):
    new_diam = math.floor((low_diam + high_diam)/4) * 2
    new_mask_indices = move_circle_to_coords(diam_to_indices[new_diam], center, com)
    new_sum = response[new_mask_indices[0], new_mask_indices[1]].sum()
    if new_sum <= energy_target:
        return new_diam, new_diam, high_diam, new_sum
    else:
        return low_diam, new_diam, new_diam, new_sum


def calculate_mask(center, diam, img_size):
    xx, yy = np.mgrid[:img_size, :img_size]
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    filled = (circle <= (diam/2) ** 2)
    return filled


def move_circle_to_coords(indices, center, coords):
    x_coords = indices[0] + coords[0] - center[0]
    y_coords = indices[1] + coords[1] - center[1]
    valid_mask = (x_coords >= 0) & (x_coords < center[0]) & (y_coords >= 0) & (y_coords < center[1])

    return x_coords[valid_mask], y_coords[valid_mask]


def generate_masks(center, img_size):
    diam_to_indices = {}
    for i in range(0, 2*int(img_size*math.sqrt(2))+1):
        mask = calculate_mask(center, i, 2*img_size)
        diam_to_indices[i] = np.where(mask)
    return diam_to_indices
