import argparse
import os
import random
import tarfile
import tempfile
import zipfile

import h5py
import matplotlib.pyplot as plt
import rarfile
import torch
import torchio as tio
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def main():
    args = create_argparser().parse_args()

    h5file = h5py.File(args.output, "w")
    h5train = h5file.create_group("train")
    h5validate = h5file.create_group("validate")
    h5test = h5file.create_group("test")

    # BRATS
    brats_subj = [
        l
        for l in os.listdir(f"{args.brats_input}")
        if not l.endswith(".csv") and "355" not in l
    ]
    brats_subj_train = brats_subj[: int((len(brats_subj) * split[0]))]
    brats_subj_validate = brats_subj[
        int((len(brats_subj) * split[0])) : int(
            (len(brats_subj) * (split[0] + split[1]))
        )
    ]
    brats_subj_test = brats_subj[int((len(brats_subj) * (split[0] + split[1]))) :]
    process_split_brats(brats_subj_train, h5train, args)
    process_split_brats(brats_subj_validate, h5validate, args)
    process_split_brats(brats_subj_test, h5test, args)

    # NFBS
    nfbs_tf = tarfile.open(args.nfbs_input, mode="r")
    nfbs_subj = [
        m.name.split("/")[1]
        for m in nfbs_tf.getmembers()
        if m.name.endswith("T1w.nii.gz")
    ]
    nfbs_subj_train = nfbs_subj[: int(len(nfbs_subj) * split[0])]
    nfbs_subj_validate = nfbs_subj[
        int(len(nfbs_subj) * split[0]) : int(len(nfbs_subj) * (split[0] + split[1]))
    ]
    nfbs_subj_test = nfbs_subj[int(len(nfbs_subj) * (split[0] + split[1])) :]
    process_split_nfbs(nfbs_tf, nfbs_subj_train, h5train, args)
    process_split_nfbs(nfbs_tf, nfbs_subj_validate, h5validate, args)
    process_split_nfbs(nfbs_tf, nfbs_subj_test, h5test, args)


def process_split_brats(subjects: list, h5group: h5py.Group, args):
    transform = tio.transforms.Compose(
        [
            tio.transforms.ToCanonical(),
            tio.CropOrPad(256),
            tio.transforms.RescaleIntensity(),
        ]
    )

    if "brain" not in h5group.keys():
        brain_group = h5group.create_group("brain")
        brain_group.create_group("healthy")
        brain_group.create_group("malignant")

    for subject in tqdm(subjects):
        sub = tio.Subject(
            image=tio.ScalarImage(f"{args.brats_input}/{subject}/{subject}_t1.nii"),
            seg=tio.LabelMap(f"{args.brats_input}/{subject}/{subject}_seg.nii"),
        )
        sub = transform(sub)

        mri_data = sub["image"].data
        mri_seg = sub["seg"].data

        for i in range(mri_data.shape[-1]):
            slice_data = mri_data[..., i]
            slice_seg = mri_seg[..., i]

            if torch.sum((slice_data > 0).float()) / (256**2) < 0.05:
                continue

            label = "malignant" if torch.max(slice_seg) > 0 else "healthy"
            # if label == "malignant" and torch.sum((slice_seg > 0).float()) < 10:
            #     continue
            slice_group = h5group["brain"][label].create_group(f"BRATS_{subject}_{i}")
            slice_group.create_dataset(name="scan", data=slice_data, compression="gzip")
            slice_group.create_dataset(
                name="seg", data=slice_seg, compression="gzip", dtype=float
            )


def process_split_nfbs(
    nfbs_tf: tarfile.TarFile, subjects: list, h5group: h5py.Group, args
):
    transform = tio.transforms.Compose(
        [
            tio.transforms.ToCanonical(),
            tio.CropOrPad(256),
            tio.transforms.RescaleIntensity(),
        ]
    )

    tmpdir = tempfile.TemporaryDirectory()
    if "brain" not in h5group.keys():
        brain_group = h5group.create_group("brain")
        brain_group.create_group("healthy")

    for subject in tqdm(subjects):
        scan_tf_member = nfbs_tf.getmember(
            f"NFBS_Dataset/{subject}/sub-{subject}_ses-NFB3_T1w_brain.nii.gz"
        )
        nfbs_tf.extract(scan_tf_member, tmpdir.name)
        sub = tio.Subject(image=tio.ScalarImage(f"{tmpdir.name}/{scan_tf_member.name}"))
        sub = transform(sub)

        mri_data = sub["image"].data

        for i in range(mri_data.shape[-1]):
            slice_data = mri_data[..., i]

            if torch.sum((slice_data > 0).float()) / 256**2 < 0.05:
                continue

            slice_group = h5group["brain"]["healthy"].create_group(
                f"NFBS_{subject}_{i}"
            )
            slice_group.create_dataset(name="scan", data=slice_data, compression="gzip")
            slice_group.create_dataset(
                name="seg",
                data=torch.zeros_like(slice_data),
                compression="gzip",
                dtype=float,
            )


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--output", type=str, default="hdf5_files/mri_combined.hdf5"
    )
    parser.add_argument("-B", "--brats_input", type=str)
    parser.add_argument("-N", "--nfbs_input", type=str)

    return parser


if __name__ == "__main__":
    random.seed(42)
    split = (0.8, 0.1, 0.1)
    main()
