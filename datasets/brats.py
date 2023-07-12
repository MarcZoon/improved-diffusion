import argparse
import os
import random

import h5py
import matplotlib.pyplot as plt
import nibabel
import torch
import torchio as tio
import tqdm


def main():
    args = create_argparser().parse_args()

    # Create HDF5 file
    h5file = h5py.File(args.output_file, "w")
    h5train = h5file.create_group("train")
    h5validate = h5file.create_group("validate")
    h5test = h5file.create_group("test")

    # Create splits
    subjects = [
        l
        for l in os.listdir(f"{args.input_dir}")
        if not l.endswith(".csv") and "355" not in l
    ]
    random.shuffle(subjects)

    subj_train = subjects[: int((len(subjects) * data_split[0]))]
    subj_validate = subjects[
        int((len(subjects) * data_split[0])) : int(
            (len(subjects) * (data_split[0] + data_split[1]))
        )
    ]
    subj_test = subjects[int((len(subjects) * (data_split[0] + data_split[1]))) :]

    process_split(subj_train, h5train, args)
    process_split(subj_validate, h5validate, args)
    process_split(subj_test, h5test, args)


def process_split(subjects: list, h5group: h5py.Group, args):
    transforms = tio.transforms.Compose(
        [
            tio.transforms.ToCanonical(),
            tio.CropOrPad(256),
            tio.transforms.RescaleIntensity(),
        ]
    )
    for subject in tqdm.tqdm(subjects):
        tiosub = tio.Subject(
            image=tio.ScalarImage(
                f"{args.input_dir}/{subject}/{subject}_{args.sequence}.nii"
            ),
            seg=tio.LabelMap(f"{args.input_dir}/{subject}/{subject}_seg.nii"),
        )
        tiosub = transforms(tiosub)

        image = tiosub["image"].data
        seg = tiosub["seg"].data

        h5sub = h5group.create_group(subject)
        h5sub.create_dataset("image", data=image / torch.max(image), compression="gzip")
        h5sub.create_dataset("label", data=seg, compression="gzip")

        # # Get slice ids for axial view where area > 0.05
        valid_axial = [
            id
            for id in range(image.shape[-1])
            if torch.sum((image[..., id] > 0).float()) >= 0.05
        ]

        healthy_axial = [
            id
            for id in range(image.shape[-1])
            if torch.sum((image[..., id] > 0).float()) >= 0.05
            and seg[:, :, id].max() == 0.0
        ]

        h5sub.create_dataset("healthy_axial", data=healthy_axial)
        h5sub.create_dataset("valid_axial", data=valid_axial)


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--sequence", default="t1", type=str)
    return parser


if __name__ == "__main__":
    random.seed(42)
    data_split = (0.8, 0.1, 0.1)
    main()
