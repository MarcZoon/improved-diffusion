import h5py
import torch
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    file_path: str,
    batch_size: int,
    split: str = "train",
    label: str = "benign",
    class_cond: bool = False,
    deterministic: bool = False
):
    if not file_path:
        raise ValueError("unspecified data_file")
    assert split in ["train", "validate", "test"]
    assert label in ["benign", "malignant", "both"]

    if USE_MPI:
        h5file = h5py.File(file_path, driver="mpio", comm=MPI.COMM_WORLD)
    else:
        h5file = h5py.File(file_path)

    if label == "both":
        ids = [(split, "benign", id) for id in h5file[split]["benign"].keys()]
        ids += [(split, "malignant", id) for id in h5file[split]["malignant"].keys()]
    else:
        ids = [(split, label, id) for id in h5file[split][label].keys()]

    dataset = ImageDataset(h5file, ids)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(
        self,
        h5file: h5py.File,
        ids: list,
        class_cond=False,
        shard=0,
        num_shards=1,
    ) -> None:
        super().__init__()
        self.h5file = h5file
        self.ids = ids[shard:][::num_shards]
        self.class_cond = class_cond

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        split, label, img_id = self.ids[idx]
        data = torch.tensor(self.h5file[split][label][img_id][...])
        cond = {"y": label} if self.class_cond else {}
        return data, cond


if __name__ == "__main__":
    USE_MPI = False
    import matplotlib.pyplot as plt

    data = load_data(
        file_path="hdf5_files/mammogram2.hdf5",
        batch_size=16,
        split="train",
        label="benign",
    )

    while True:
        img, _ = next(data)

        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(img[i, 0, ...], cmap="gray")
        plt.show()

else:
    from mpi4py import MPI

    USE_MPI = True
