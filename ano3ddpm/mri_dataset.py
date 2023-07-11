import os
import random
from typing import Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_data(
    *,
    file_path: str,
    batch_size: int,
    split: str,
    deterministic: bool = False,
    random_slice: bool = False,
    transform=None,
):
    if not file_path:
        raise ValueError("unspecified file path")

    if USE_MPI:
        print("using MPI")
        h5file = h5py.File(file_path, mode="r", driver="mpio", comm=MPI.COMM_WORLD)
    else:
        h5file = h5py.File(file_path, mode="r")

    if transform is None:
        transform_list = [
            transforms.ToPILImage(),
        ]
        if split == "train":
            transform_list.append(transforms.RandomAffine(3, (0.02, 0.09)))
        transform_list += [
            transforms.CenterCrop(235),
            transforms.Resize((256, 256), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
        transform = transforms.Compose(transform_list)

    dataset = HDF5Dataset(
        data=h5file[split],
        random_slice=random_slice,
        shard=MPI.COMM_WORLD.Get_rank() if USE_MPI else 0,
        num_shards=MPI.COMM_WORLD.Get_size() if USE_MPI else 1,
        transform=transform,
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
        data: Union[h5py.File, h5py.Group],
        random_slice: bool = False,
        shard: int = 0,
        num_shards: int = 1,
        transform=None,
        img_only=True,
    ) -> None:
        super().__init__()

        self.data = data
        subjects = list(self.data.keys())
        self.local_subjects = subjects[shard:][::num_shards]
        self.random = random_slice
        self.transform = transform if transform else None
        self.img_only = img_only

    def __len__(self):
        return len(self.local_subjects)

    def __getitem__(self, idx):
        subject = self.data[self.local_subjects[idx]]
        image = torch.tensor(subject["image"][...])

        label = subject["label"][...]
        if self.random:
            slice_id = random.choice(subject["healthy_axial"])
            image = image[..., slice_id]
            label = label[..., slice_id]

        imgdata = {}
        if self.transform:
            imgdata["original"] = image
            image = self.transform(image)

        imgdata["image"] = image
        imgdata["label"] = label

        if self.img_only:
            return image, {}
        return imgdata


if __name__ == "__main__":
    USE_MPI = False
    data = load_data(
        file_path="./hdf5_files/BraTS2020.hdf5",
        batch_size=8,
        split="train",
        random_slice=True,
    )
    batch, cond = next(data)
    print(batch.shape)
else:
    from mpi4py import MPI

    USE_MPI = True
