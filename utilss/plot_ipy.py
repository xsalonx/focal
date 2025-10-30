from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import typing

def responses_slider(
        responses,
        true_hit_coords = None,
        found_hit_coords = None,
        title='Reponses Slider',
        figsize=(3, 3)
        ):
    # Function to display images
    def show_image(index):
        fig, ax = plt.subplots(figsize=figsize)
        img = responses[index]
        im = ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax)

        if true_hit_coords is not None:
            ax.scatter(*true_hit_coords[index], c='red', s=25)
        if found_hit_coords is not None:
            ax.scatter(*found_hit_coords[index], c='blue', s=25)
        plt.show()

    # Create slider
    # interact(show_image, index=(0, responses_with_no_nans_mask.sum() - 1));
    return interact(show_image, index=(0, len(responses) - 1));


def responses_slider_with_energy(
        responses,
        true_hit_coords = None,
        found_hit_coords = None,
        energy_distribution = None,
        title='Reponses Slider',
        figsize=(3, 3)
        ):
    # Function to display images
    def show_image(index):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        img = responses[index]
        im = ax1.imshow(img)
        ax1.set_title(title)
        ax1.axis('off')
        fig.colorbar(im, ax=ax1)

        if true_hit_coords is not None:
            ax1.scatter(*true_hit_coords[index], c='red', s=25)
        if found_hit_coords is not None:
            ax1.scatter(*found_hit_coords[index], c='blue', s=25)
    
        ax2.scatter(range(len(energy_distribution[index])), energy_distribution[index])
        ax2.set_ylabel('Cumultative energy')
        ax2.set_xlabel('Radius')
        plt.show()

    # Create slider
    # interact(show_image, index=(0, responses_with_no_nans_mask.sum() - 1));
    return interact(show_image, index=(0, len(responses) - 1));
