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


def newmain():
    args = create_argparser().parse_args()

    # Create HDF5 file
    h5file = h5py.File(args.output_file, "w")
    h5train = h5file.create_group("train")
    h5validate = h5file.create_group("validate")
    h5test = h5file.create_group("test")

    tf = tarfile.open(args.input_file, mode="r")
    subs = [
        m.name.split("/")[1] for m in tf.getmembers() if m.name.endswith("T1w.nii.gz")
    ]
    random.shuffle(subs)

    subs_train = subs[: int(len(subs) * split[0])]
    subs_validate = subs[
        int(len(subs) * split[0]) : int(len(subs) * (split[0] + split[1]))
    ]
    subs_test = subs[int(len(subs) * (split[0] + split[1])) :]

    newprocess_split(tf, subs_train, h5train, args)
    newprocess_split(tf, subs_validate, h5validate, args)
    newprocess_split(tf, subs_test, h5test, args)


def newprocess_split(tf: tarfile.TarFile, subs: list, h5group: h5py.Group, args):
    transforms = tio.transforms.Compose(
        [
            tio.transforms.ToCanonical(),
            tio.CropOrPad(256),
            tio.transforms.RescaleIntensity(),
        ]
    )

    for sub in tqdm(subs):
        img = tf.getmember(
            f"NFBS_Dataset/{sub}/sub-{sub}_ses-NFB3_T1w{'_brain' if args.brain else ''}.nii.gz"
        )
        tf.extract(img, tmp)
        mask = tf.getmember(
            f"NFBS_Dataset/{sub}/sub-{sub}_ses-NFB3_T1w_brainmask.nii.gz"
        )
        tf.extract(mask, tmp)
        subject = tio.Subject(
            image=tio.ScalarImage(f"{tmp}/{img.name}"),
            mask=tio.LabelMap(f"{tmp}/{mask.name}"),
        )
        subject = transforms(subject)
        subject.add_image(
            image=tio.LabelMap(tensor=torch.zeros_like(subject["image"].data)),
            image_name="label",
        )

        valid_indices = [
            i
            for i in range(subject["mask"].data.shape[-1])
            if torch.sum((subject["mask"].data[..., i] > 0).float()) / (256**2)
            >= 0.05
        ]

        subgroup = h5group.create_group(sub)
        subgroup.create_dataset("image", data=subject["image"].data, compression="gzip")
        subgroup.create_dataset("label", data=subject["label"].data, compression="gzip")
        subgroup.create_dataset("valid_axial", data=valid_indices, compression="gzip")
        subgroup.create_dataset("healthy_axial", data=valid_indices, compression="gzip")


def process_split(tf: tarfile.TarFile, members: list, h5group: h5py.Group):
    transforms = tio.transforms.Compose(
        [
            tio.transforms.ToCanonical(),
            tio.CropOrPad(256),
            tio.transforms.RescaleIntensity(),
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
        subgroup.create_dataset("valid_axial", data=indices, compression="gzip")


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--brain", default=False, type=bool)
    return parser


if __name__ == "__main__":
    random.seed(42)
    tmp = tempfile.mkdtemp()
    split = (0.8, 0.1, 0.1)
    newmain()
