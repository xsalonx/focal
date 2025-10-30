import numpy as np
from utilss.image_analyzis.approx_radius import IDX_LIST
from utilss.data_description import RESPONSE_SHAPE

def get_energy_characteristics(response, centroid):
    energies_stack = []
    energy_sum = 0

    for r, idx in enumerate(IDX_LIST):
        x_coords = idx[0] + int(centroid[0])
        y_coords = idx[1] + int(centroid[1])

        valid_mask = (x_coords >= 0) & (x_coords < RESPONSE_SHAPE[0]) & (y_coords >= 0) & (y_coords < RESPONSE_SHAPE[1])

        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]

        energy_sum += response[y_valid, x_valid].sum()
        energies_stack.append(energy_sum)

    return np.array(energies_stack)

def get_image_from_characteristics(centroid, characteristic):
    image = np.zeros(RESPONSE_SHAPE)

    x, y = int(centroid[0]), int(centroid[1]) 

    # print('CH shape', characteristic)
    image[y, x] = characteristic[0]

    # for r, idx in enumerate(IDX_LIST):
    for radius in range(1, len(IDX_LIST)):
        idx = IDX_LIST[radius]
        x_coords = idx[0] + x
        y_coords = idx[1] + y

        valid_mask = (x_coords >= 0) & (x_coords < RESPONSE_SHAPE[0]) & \
            (y_coords >= 0) & (y_coords < RESPONSE_SHAPE[1]) & \
            np.logical_or(
                np.logical_or(x_coords < 7 * 7, x_coords >= 7 * 8),
                np.logical_or(y_coords < 7 * 7, y_coords >= 7 * 8))
            

        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]

        cells_no = x_valid.shape[0]
        if cells_no == 0 and radius > 15:
            break
        if cells_no > 0:
            cell_value = (characteristic[radius] - characteristic[radius-1]) / cells_no
        else:
            cell_value = 0

        image[y_valid, x_valid] = cell_value

    return image