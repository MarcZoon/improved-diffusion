import os
import random
from typing import Union

import h5py
import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_data(
    *,
    file_path: str,
    batch_size: int,
    split: str,
    organs: Union[str, list] = "all",
    labels: str = "healthy",
    deterministic: bool = False,
    transform=None,
    class_cond=False,
):
    if not file_path:
        raise ValueError("unspecified file path")

    if USE_MPI:
        h5file = h5py.File(file_path, mode="r", driver="mpio", comm=MPI.COMM_WORLD)
    else:
        h5file = h5py.File(file_path, mode="r")

    if transform is None:
        transform_list = [
            transforms.ToPILImage(),
        ]
        if split == "train":
            transform_list.append(transforms.RandomAffine(3, (0.02, 0.09)))
            # transform_list.append(tio.transforms.RandomAffine(degrees=3, translation=(0.02, 0.09)))
        transform_list += [
            transforms.CenterCrop(235),
            transforms.Resize((256, 256), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
        transform = transforms.Compose(transform_list)

    dataset = HDF5Dataset(
        hdf5file=h5file,
        split=split,
        organs=organs,
        labels=labels,
        shard=MPI.COMM_WORLD.Get_rank() if USE_MPI else 0,
        num_shards=MPI.COMM_WORLD.Get_size() if USE_MPI else 1,
        transform=transform,
        class_cond=class_cond,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
    )
    while True:
        yield from loader


class HDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5file: h5py.File,
        split: str,
        organs: Union[str, list],
        labels: str,
        shard: int = 0,
        num_shards: int = 1,
        transform=None,
        class_cond: bool = False,
    ) -> None:
        super().__init__()

        self.h5file = hdf5file
        self.split = split
        self.class_cond = class_cond
        self.transform = transform

        if organs == "all":
            self.organs = list(hdf5file[split].keys())
        elif isinstance(organs, str):
            self.organs = organs.split(",")
        else:
            self.organs = organs

        self.labels = labels.split(",")

        self.items = []
        for organ in self.organs:
            for label in self.h5file[split][organ].keys():
                if label in labels or "all" in labels:
                    slices = [s for s in self.h5file[split][organ][label].keys()]
                    for s in slices:
                        self.items.append((organ, label, s))

        self.local_items = self.items[shard:][::num_shards]

        self.classes = {x: i for i, x in enumerate(sorted(set(self.organs)))}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        organ, label, scan_id = self.items[idx]

        scan_data = self.h5file[self.split][organ][label][f"{scan_id}"]["scan"][...]
        scan_seg = self.h5file[self.split][organ][label][f"{scan_id}"]["seg"][...]

        out_dict = {
            "seg": scan_seg,
            "original": scan_data,
        }
        if self.class_cond:
            out_dict["y"] = self.classes[organ]

        img = self.transform(scan_data) if self.transform is not None else scan_data
        return img, out_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    USE_MPI = False
    data = load_data(
        file_path="./hdf5_files/mri_combined.hdf5",
        batch_size=2,
        split="train",
        organs="all",
        labels="all",
        class_cond=False,
    )

    while True:
        b, out_dict = next(data)
        print(out_dict)
        plt.subplot(1, 2, 1)
        plt.imshow(b[0, 0, ...], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(out_dict["seg"][0, 0, ...], cmap="gray")
        plt.show()

else:
    from mpi4py import MPI

    USE_MPI = True
