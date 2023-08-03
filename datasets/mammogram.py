import argparse
import random
import tempfile
import zipfile

import h5py
import numpy as np
import rarfile
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def main():
    args = create_argparser().parse_args()

    h5file = h5py.File(args.output_file, "w")
    h5train = h5file.create_group("train")
    h5validate = h5file.create_group("validate")
    h5test = h5file.create_group("test")

    zip_file = zipfile.ZipFile(args.input_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        rar_path = zip_file.extract(zip_file.filelist[0], tmpdir)
        rar_file = rarfile.RarFile(rar_path)
        benign_files = [
            file
            for file in rar_file.namelist()
            if "INbreast+MIAS+DDSM Dataset/Benign Masses" in file
            and file.endswith(".png")
        ]
        malignant_files = [
            file
            for file in rar_file.namelist()
            if "INbreast+MIAS+DDSM Dataset/Malignant Masses" in file
            and file.endswith(".png")
        ]

        # benign splits
        train_benign = benign_files[: int((len(benign_files) * data_split[0]))]
        validate_benign = benign_files[
            int((len(benign_files) * data_split[0])) : int(
                (len(benign_files) * (data_split[0] + data_split[1]))
            )
        ]
        test_benign = benign_files[
            int((len(benign_files) * (data_split[0] + data_split[1]))) :
        ]

        # malignant splits
        train_malignant = malignant_files[: int((len(malignant_files) * data_split[0]))]
        validate_malignant = malignant_files[
            int((len(malignant_files) * data_split[0])) : int(
                (len(malignant_files) * (data_split[0] + data_split[1]))
            )
        ]
        test_malignant = malignant_files[
            int((len(malignant_files) * (data_split[0] + data_split[1]))) :
        ]

        process_split(rar_file, train_benign, train_malignant, h5train)
        process_split(rar_file, validate_benign, validate_malignant, h5validate)
        process_split(rar_file, test_benign, test_malignant, h5test)


def process_split(
    rar_file: rarfile.RarFile,
    benign_files: list,
    malignant_files: list,
    h5group: h5py.Group,
):
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            torch.nn.ZeroPad2d([15, 14, 15, 14]),
        ]
    )
    h5benign = h5group.create_group("benign")
    h5malignant = h5group.create_group("malignant")

    for file in tqdm(benign_files):
        with tempfile.TemporaryDirectory() as tmpdir:
            png = rar_file.extract(file, tmpdir)
            img_id = png.split("/")[-1].removesuffix(".png")
            img = Image.open(png)
            tensor = transform(img)
            tensor = tensor.float()
            tensor = tensor / torch.max(tensor)
            h5benign.create_dataset(img_id, data=tensor, compression="gzip")
    for file in tqdm(malignant_files):
        with tempfile.TemporaryDirectory() as tmpdir:
            png = rar_file.extract(file, tmpdir)
            img_id = png.split("/")[-1].removesuffix(".png")
            img = Image.open(png)
            tensor = transform(img)
            tensor = tensor.float()
            tensor = tensor / torch.max(tensor)
            h5malignant.create_dataset(img_id, data=tensor, compression="gzip")


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file")
    parser.add_argument("--output_file", default="hdf5_files/mammogram.hdf5")
    return parser


if __name__ == "__main__":
    random.seed(42)
    data_split = (0.8, 0.1, 0.1)
    main()
