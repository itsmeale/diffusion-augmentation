import pandas as pd
import torch
import torchvision as tvis
from PIL import Image
from torch.utils.data import Dataset

MHIST_IMAGE_PATH: str = "data/mhist/images/{image_name}"


preprocess = tvis.transforms.Compose(
    [
        tvis.transforms.ToTensor(),
        tvis.transforms.Normalize((0.5), (0.5)),
    ]
)


def get_annotations(partition: str) -> pd.DataFrame:
    # partition should be in ["train", "test"]

    if partition not in {"train", "test"}:
        raise Exception("Partition don't exist.")

    usecols = [
        "Image Name",
        "Majority Vote Label",
        "Partition",
    ]

    df = pd.read_csv("data/mhist/annotations.csv", usecols=usecols).dropna()
    df = df[df.Partition == partition]
    df.columns = ["image_name", "target_label", "partition"]

    possible_targets = df["target_label"].unique()

    map_targets = {
        target: encoded_target
        for target, encoded_target in zip(
            sorted(possible_targets), range(len(possible_targets))
        )
    }
    df["target"] = df["target_label"].map(map_targets)

    df["image_path"] = df.apply(
        lambda r: MHIST_IMAGE_PATH.format(image_name=r["image_name"]), axis=1
    )

    return df[["image_path", "target"]].values


class MHISTDataset(Dataset):
    def __init__(self, partition: str):
        super(MHISTDataset, self).__init__()
        self.annotations = get_annotations(partition=partition)
        self.dataset = list()
        self.load_dataset()

    def load_dataset(self):
        for im_path, label in self.annotations:
            self.dataset.append(
                (
                    preprocess(Image.open(im_path)),
                    torch.as_tensor(label).type(torch.FloatTensor),
                )
            )
        return self

    def to_cuda(self):
        if torch.cuda.is_available():
            cuda_images = [(im.to("cuda"), c.to("cuda")) for im, c in self.dataset]
            self.dataset = cuda_images
        return self

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.annotations)
