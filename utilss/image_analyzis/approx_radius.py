import numpy as np

H, W = 105, 105

# list of indices for a circle of given radius. Only the circumference line
# indices are centered at 0 so later on we can just add the center coordinates
IDX_LIST = []

# sqrt because if the center is in the corner we want to reach the other corner
for i in range(np.sqrt(H**2 + W**2).astype(int)):
    value_range = np.arange(-i, i + 1)
    col = (np.arange(-i, i + 1) ** 2)[:, None]
    row = np.arange(-i, i + 1) ** 2

    # conditions here ensure that we use every pixel once
    radius = (col + row >= i**2) & (col + row < (i + 1) ** 2)
    radius_indices = np.where(radius)

    IDX_LIST.append((value_range[radius_indices[0]], value_range[radius_indices[1]]))

    
