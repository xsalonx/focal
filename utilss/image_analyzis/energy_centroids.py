from skimage import filters, measure, morphology
from scipy.ndimage import gaussian_filter
from utils.data_description import RESPONSE_SHAPE
from utils.image_analyzis.approx_radius import IDX_LIST
import numpy as np


def find_centroids(response):
    response_scaled = np.log(response + 1e-10)
    response_scaled[response_scaled < 1e-3] = 0
    smoothed = gaussian_filter(response_scaled, sigma=1)
    threshold = filters.threshold_otsu(smoothed)
    binary = smoothed > threshold
    binary_cleaned = morphology.remove_small_objects(binary, min_size=5)
#     binary_cleaned = morphology.binary_closing(binary_cleaned, morphology.disk(10))
    labels = measure.label(binary_cleaned)
    regions = measure.regionprops(labels)
    centroids = [region.centroid for region in regions]
    return centroids



# @deprecated
# def get_energies_from_centroid(response, centroids, target_centroid_id):
#     energies_stack = []
#     centroid = centroids[target_centroid_id]
#     energy_sum = 0
#     for r, idx in enumerate(IDX_LIST):
#         x_coords = idx[0] + int(centroid[0])
#         y_coords = idx[1] + int(centroid[1])

#         valid_mask = (x_coords >= 0) & (x_coords < RESPONSE_SHAPE[0]) & (y_coords >= 0) & (y_coords < RESPONSE_SHAPE[1])

#         x_valid = x_coords[valid_mask]
#         y_valid = y_coords[valid_mask]

#         energy_sum += response[x_valid, y_valid].sum()
#         energies_stack.append(energy_sum)
#         coors = list(zip(x_valid, y_valid))
#         break_main_loop = False
#         impeding_centorid_id = None
#         for other_id, other_centroid in enumerate(centroids):
#             if other_id != target_centroid_id:
#                 rounded_other_centroid = (int(other_centroid[0]), int(other_centroid[1]))
#                 if rounded_other_centroid in coors:
#                     break_main_loop = True
#                     impeding_centorid_id= other_id
#                     break

#         if break_main_loop:
#             break

#     return np.array(energies_stack), impeding_centorid_id