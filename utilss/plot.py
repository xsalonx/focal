import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import typing

def plot_responses_rows(
        images,
        true_coordinates=None,
        found_cordinates=None,
        labels: typing.Sequence[str] | None = None,
        colorbar=True,
        title=None,
        clip_negative = False,
        size_scaler=1,
    ):
    """Plot one or many groups of images.

    Parameters
    ----------
    images: array-like of image arrays
        Sequence containing arrays of images. Each array should have shape
        ``(N, ...)`` where ``N`` is the number of images.
    step: int
        Current training step.
    labels: Sequence[str], optional
        Row labels for each group of images.
    """

    # Accept single array or list/tuple of arrays
    if not isinstance(images, (list, tuple)):
        images = [images]

    # number of images to show per group; never exceed images_in_row
    n = min(*[il.shape[0] for il in images], 10000000000000000000)

    total_rows = len(images)
    fig, axs = plt.subplots(total_rows, n,
                            figsize=((2 * n + 1) * size_scaler, (2 * total_rows) * size_scaler),
                            dpi=200)
    
    if title is not None:
        fig.suptitle(title)

    if total_rows == 1:
        axs = axs.reshape((1, n))
    elif n == 1:
        axs = axs.reshape((total_rows, 1))

    for group_idx, imgs in enumerate(images):
        m = min(n, imgs.shape[0])
        for i in range(m):
            x = imgs[i]
            x = x if not clip_negative else np.clip(x, 0, x.max())
            ax = axs[group_idx, i]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            im = ax.imshow(x, interpolation='none', cmap='gnuplot')
            if colorbar:
                fig.colorbar(im, ax=ax)

            hit_xy = true_coordinates[i] if true_coordinates is not None else None
            if hit_xy is not None:
                ax.scatter(*hit_xy, c='red', s=24)

            hit_xy = found_cordinates[i] if found_cordinates is not None else None
            if hit_xy is not None:
                ax.scatter(*hit_xy, c='blue', s=24)


    # Add labels on the left side of each group of rows
    if labels is None:
        labels = [f'Group {i + 1}' for i in range(len(images))]

    for group_idx, label in enumerate(labels):
        axs[group_idx, 0].set_ylabel(label)

    plt.tight_layout()

    return fig, axs


def hist_fast(ax, x, bins, use_log=False, density=False, **kwargs):
    """Fast histogram plotting. More than 20x faster than `ax.hist`
    
    :param ax: Matplotlib Axes object.
    :param x: Array with data.
    :param density: See np.histogram.
    :param kwargs: Extra keyword-args passed to `ax.fill_between`
    """
    # Calculate histogram bins and edges.
    hist, bin_edges = np.histogram(x, bins=bins, density=density)
    if use_log:
        hist = np.log(hist + 1e-10)
    
    # Repeat histogram bins and edges to create steps.
    hist_steps = np.repeat(hist, 2)
    bin_edges_steps = np.repeat(bin_edges, 2)[1:-1]

    # Plot solid color as histogram.
    ax.fill_between(bin_edges_steps, 0.0, hist_steps, **kwargs)
    
    # Adjust y-axis limits.
    _, y_max = ax.get_ylim()
    ax.set_ylim(0.0, y_max)


def responses_hists(features, responses, DAT_DIR, figsize=(37, 4), bins=500, fontsize=12, file_name='plots'):
    fig, axs = plt.subplots(1, 4, figsize=figsize)
    fontdict = { 'fontsize': fontsize }

    # histograms
    cells_flatten = responses.flatten()
    # hist_fast(axs[0], cells_flatten, bins=bins)
    # axs[0].set_title('Cells energy distribution', fontdict=fontdict)
    # axs[0].set_xlabel('Cell energy')

    hist_fast(axs[0], cells_flatten, bins=bins, use_log=True)
    axs[0].set_title('log of cells energy\ndistribution', fontdict=fontdict)
    axs[0].set_xlabel('Cell energy', fontdict=fontdict)

    total_flatten = responses.sum(axis=(1, 2)).flatten()
    hist_fast(axs[1], total_flatten, bins=bins)
    axs[1].set_title('Total response energy\ndistribution', fontdict=fontdict)
    axs[1].set_xlabel('Response energy', fontdict=fontdict)
    
    # hist_fast(axs[3], total_flatten, bins=bins, use_log=True)
    # axs[3].set_title('log of Total response energy distribution', fontdict=fontdict)
    # axs[3].set_xlabel('Response enregy')

    # other plots
    particlesEnergy = features[:, 0]
    axs[2].scatter(particlesEnergy, total_flatten, s=0.08)
    axs[2].set_title('Total energy vs particle energy', fontdict=fontdict)
    axs[2].set_xlabel('Particle Energy', fontdict=fontdict)
    axs[2].set_ylabel('Total response energy', fontdict=fontdict)

    axs[3].scatter(particlesEnergy, total_flatten / particlesEnergy, s=0.08)
    axs[3].set_title('Energy ratio (total/particle)\nvs particle energy', fontdict=fontdict)
    axs[3].set_xlabel('Particle Energy', fontdict=fontdict)
    axs[3].set_ylabel('Energy Ratio', fontdict=fontdict)

    # # avg. image vis
    # axs[6].imshow(responses.sum(axis=0) / responses.shape[0])
    # axs[6].set_title('Avg. Response', fontdict=fontdict)

    plt.tight_layout()

    fig.savefig(DAT_DIR / f'{file_name}.png')



def show_image_grid(responses, num_images=16, nrow=4, padding=2, title="Dataset Samples", figsize=(10, 10), data_dir=None):
    images = torch.Tensor(responses[:num_images]).unsqueeze(1)
    images_grid = vutils.make_grid(images, nrow=nrow, padding=padding, pad_value=255)
    np_grid = images_grid.permute(1, 2, 0).numpy()

    fig = plt.figure(figsize=figsize)
    plt.imshow(np_grid, cmap='gnuplot')
    plt.title(title)
    plt.axis("off")
    plt.show()
    if data_dir is not None:
        fig.savefig(data_dir / (title.lower().replace(' ', '-') + '-plot.png'))