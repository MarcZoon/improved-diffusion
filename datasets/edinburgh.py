import argparse
import os
import random
import tarfile
import tempfile
import zipfile

import h5py
import numpy as np
import torchio as tio
from tqdm import tqdm


def main():
    args = create_argparser().parse_args()

    h5file = h5py.File(args.output_file, "w")
    h5train = h5file.create_group("train")
    h5validate = h5file.create_group("validate")
    h5test = h5file.create_group("test")

    tarfiles = [
        os.path.join(args.input_dir, file)
        for file in os.listdir(args.input_dir)
        if file.endswith(".tar")
    ]
    random.shuffle(tarfiles)

    train_files = tarfiles[: int((len(tarfiles) * data_split[0]))]
    validate_files = tarfiles[
        int((len(tarfiles) * data_split[0])) : int(
            (len(tarfiles) * (data_split[0] + data_split[1]))
        )
    ]
    test_files = tarfiles[int((len(tarfiles) * (data_split[0] + data_split[1]))) :]

    process_split(train_files, h5train)
    # process_split(validate_files, h5validate)
    # process_split(test_files, h5test)


def process_split(files: list, h5group: h5py.Group):
    transforms = tio.transforms.Compose(
        [
            tio.transforms.ToCanonical(),
            tio.CropOrPad(256),
            tio.transforms.RescaleIntensity(),
        ]
    )

    for file in tqdm(files):
        tar_file = tarfile.TarFile(file, "r")
        tar_info = [
            member
            for member in tar_file.getmembers()
            if "COR_3D_IR_PREP" in member.name
        ][0]
        patient_id = tar_info.name.split("/")[0]
        h5sub = h5group.create_group(patient_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            tar_file.extract(tar_info, tmpdir)
            zip_file = zipfile.ZipFile(os.path.join(tmpdir, tar_info.name))

            zip_info = [
                file for file in zip_file.filelist if file.filename.endswith(".nii")
            ][0]
            nii_file = zip_file.extract(zip_info, tmpdir)

            tiosub = tio.Subject(image=tio.ScalarImage(nii_file))
            tiosub = transforms(tiosub)

            h5sub.create_dataset("image", data=tiosub["image"].data)
            h5sub.create_dataset("label", data=np.zeros_like(tiosub["image"].data))
            h5sub.create_dataset("valid_axial", data=list(range(100, 130)))
            h5sub.create_dataset("healthy_axial", data=list(range(100, 130)))


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_file", type=str, default="hdf5_files/Edinburgh.hdf5")
    return parser


if __name__ == "__main__":
    random.seed(42)
    data_split = (0.8, 0.1, 0.1)
    main()
