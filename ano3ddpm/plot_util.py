from typing import List, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from .simplex_util import generateSimplex


def save_as_plot(
    data: Union[List[torch.Tensor], List[np.ndarray]], path: str, stepsize=50
):
    start, intermediate, last = data[0], data[1:-1][::stepsize], data[-1]

    nrows = start.shape[0]
    ncols = len(intermediate) + 2
    for b in range(start.shape[0]):
        # x_start
        plt.subplot(nrows, ncols, 1 + (b * ncols))
        plt.imshow(start[b, 0, ...], cmap="gray")
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # diffusion process
        for j, i in enumerate(intermediate):
            plt.subplot(nrows, ncols, (j + 2) + (b * ncols))
            plt.imshow(i[b, 0, ...], cmap="gray")
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # result
        plt.subplot(nrows, ncols, (b + 1) * ncols)
        plt.imshow(last[b, 0, ...], cmap="gray")
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(path, dpi=600)


def save_as_video(data: Union[List[torch.Tensor], List[np.ndarray]], path: str):
    batch_size = data[0].shape[0]

    fig, axs = plt.subplots(batch_size, 1)
    if batch_size == 1:
        axs = [axs]

    plots = [axs[b].imshow(data[0][b, 0, ...], cmap="gray") for b in range(batch_size)]

    def update(t):
        for b, plot in enumerate(plots):
            plot.set_data(data[t][b, 0, ...])
        return plots

    frames = [0] * 10 + list(range(len(data))) + [len(data) - 1] * 10

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=10000 / len(data)
    )
    ani.save(path, writer="ffmpeg")


def save_evaluation_plot(
    diffusion_data: Union[List[torch.Tensor], List[np.ndarray]],
    mse: Union[torch.Tensor, np.ndarray],
    pred: Union[torch.Tensor, np.ndarray],
    segmentation: Union[torch.Tensor, np.ndarray],
    path: str,
):
    nrows = segmentation.shape[0]
    ncols = len(diffusion_data) + 3
    fig = plt.figure(figsize=(8, 8))
    for b in range(nrows):
        for i, data in enumerate(diffusion_data):
            ax = fig.add_subplot(nrows, ncols, 1 + i + (b * ncols))
            ax.imshow(data[b, 0, ...], cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")

        ax = fig.add_subplot(nrows, ncols, 1 + len(diffusion_data) + (b * ncols))
        ax.imshow(mse[b, 0, ...], cmap="gist_heat")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect("equal")

        ax = fig.add_subplot(nrows, ncols, 2 + len(diffusion_data) + (b * ncols))
        ax.imshow(pred[b, 0, ...], cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect("equal")

        ax = fig.add_subplot(nrows, ncols, 3 + len(diffusion_data) + (b * ncols))
        ax.imshow(segmentation[b, 0, ...], cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect("equal")

    fig.patch.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path, dpi=600)


def main():
    x = torch.zeros((2, 1, 64, 64))

    data = [generateSimplex(x, octaves=1) for _ in range(251)]
    # data = [generateSimplex(x, octaves=1) for _ in range(10)]
    save_as_plot(data, "./test.png")
    save_as_video(data, "./test_video.mp4")


if __name__ == "__main__":
    main()
    exit()
    # fig, ax = plt.subplots()
    # rng = np.random.default_rng(19680801)
    # data = np.array([20, 20, 20, 20])
    # x = np.array([1, 2, 3, 4])

    # artists = []
    # colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    # for i in range(20):
    #     data += rng.integers(low=0, high=10, size=data.shape)
    #     container = ax.barh(x, data, color=colors)
    #     print(type(container))
    #     artists.append(container)

    # ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    # plt.show()
