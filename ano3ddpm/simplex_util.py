from typing import List, Tuple, Union

import numpy as np
import opensimplex
import torch


def generateSimplex2D(
    shape: Union[List[int], Tuple[int]] = [256, 256],
    octaves: int = 6,
    persistance: float = 0.8,
    frequency: int = 64,
) -> np.ndarray:
    noise = np.zeros(shape)
    y, x = [np.arange(0, end) for end in shape]

    amplitude = 1.0
    for _ in range(octaves):
        opensimplex.random_seed()
        noise += amplitude * opensimplex.noise2array(x / frequency, y / frequency)
        frequency /= 2
        amplitude *= persistance

    return noise


def generateSimplex3D(
    shape: Union[List[int], Tuple[int]] = [256, 256, 256],
    octaves: int = 6,
    persistance: float = 0.8,
    frequency: int = 64,
) -> np.ndarray:
    noise = np.zeros(shape)
    z, y, x = [np.arange(0, end) for end in shape]

    amplitude = 1.0
    for _ in range(octaves):
        opensimplex.random_seed()
        noise += amplitude * opensimplex.noise3array(
            x / frequency, y / frequency, z / frequency
        )
        frequency /= 2
        amplitude *= persistance

    return noise


def generateSimplex(
    x: torch.Tensor,
    octaves: int = 6,
    persistance: float = 0.8,
    frequency: int = 64,
) -> torch.Tensor:
    batch, channels, *shape = x.shape
    dimensions = len(shape)

    noise = np.zeros(x.shape)
    for b in range(batch):
        for c in range(channels):
            if dimensions == 2:
                n = generateSimplex2D(shape, octaves, persistance, frequency)
            elif dimensions == 3:
                n = generateSimplex3D(shape, octaves, persistance, frequency)
            else:
                raise ValueError(
                    "'x' has an unsupported shape."
                    "It should be one of (B, C, x, y) or (B, C, x, y, z)"
                )
            noise[b, c, ...] = n
    return torch.tensor(noise, dtype=x.dtype, device=x.device)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    shape = (1, 1, 256, 256)
    x = torch.rand(shape)
    batch, channels, *shape = x.shape
    dimensions = len(shape)

    if dimensions == 2:
        for i in range(16):
            noise = generateSimplex(x, frequency=64)
            plt.subplot(4, 4, i + 1)
            plt.imshow(noise[0, 0, ...], cmap="gray")
    elif dimensions == 3:
        noise = generateSimplex(x, frequency=64)
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(noise[0, 0, i, ...], cmap="gray")

    plt.show()
