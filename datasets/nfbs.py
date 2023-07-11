import argparse
import random
import tarfile
import tempfile

import h5py
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm


def main():
    args = create_argparser().parse_args()

    # Create HDF5 file
    h5file = h5py.File(args.output_file, "w")
    h5train = h5file.create_group("train")
    h5validate = h5file.create_group("validate")
    h5test = h5file.create_group("test")

    tf = tarfile.open(args.input_file, mode="r")
    members = [
        i
        for i in tf.getmembers()
        if i.name.endswith(f"T1w{f'_{args.suffix}' if args.suffix else ''}.nii.gz")
    ]
    random.shuffle(members)
    members = list(members)
    members_train = members[: int(len(members) * split[0])]
    members_validate = members[
        int(len(members) * split[0]) : int(len(members) * (split[0] + split[1]))
    ]
    members_test = members[int(len(members) * (split[0] + split[1])) :]

    process_split(tf, members_train, h5train)
    process_split(tf, members_validate, h5validate)
    process_split(tf, members_test, h5test)


def process_split(tf: tarfile.TarFile, members: list, h5group: h5py.Group):
    transforms = tio.transforms.Compose(
        [
            tio.transforms.ToCanonical(),
        ]
    )
    for member in tqdm(members):
        name = member.name.split("/")[1]
        subgroup = h5group.create_group(name)
        tf.extract(member, tmp)

        image = tio.ScalarImage(f"{tmp}/{member.name}")
        image = transforms(image)

        indices = []
        for i in range(256):
            m = (image.data[0, ..., i] > 0).float()
            pct = torch.sum(m) / (256**2)
            if pct >= 0.05:
                indices.append(i)

        label = np.zeros_like(image.data)
        subgroup.create_dataset("image", data=image.data, compression="gzip")
        subgroup.create_dataset("label", data=label, compression="gzip")

        # Slice ids for axial view
        subgroup.create_dataset("healthy_axial", data=indices, compression="gzip")


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--suffix", default="", type=str)
    return parser


if __name__ == "__main__":
    random.seed(42)
    tmp = tempfile.mkdtemp()
    split = (0.8, 0.1, 0.1)
    main()
