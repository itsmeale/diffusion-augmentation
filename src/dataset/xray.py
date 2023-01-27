import gc
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision as tvis
from PIL import Image
from torch.utils.data import Dataset

IMAGE_RESOLUTION = (64, 64)


CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}


class XRayDataset(Dataset):
    def __init__(self, root_dir: str, image_resolution: int, normalize_params: Tuple):
        self.images = None
        self.files = None
        self.root_dir = root_dir
        normalize_mean, normalize_std = normalize_params
        self.preprocess = tvis.transforms.Compose(
            [
                tvis.transforms.ToTensor(),
                # tvis.transforms.Resize((image_resolution, image_resolution)),
                tvis.transforms.Grayscale(),
                tvis.transforms.Normalize((normalize_mean), (normalize_std)),
            ]
        )

        self.get_dataset_files()
        self.load_dataset()

    def get_dataset_files(self):
        root_path = Path(self.root_dir)
        classes = os.listdir(root_path)

        c_files = []
        for c in classes:
            class_path = root_path.joinpath(c)
            c_files += [
                (class_path.joinpath(f), CLASSES[c]) for f in os.listdir(class_path)
            ]

        self.files = c_files
        return self

    def load_dataset(self):
        self.images = [
            (
                self.preprocess(Image.open(path)),
                torch.as_tensor(label).type(torch.FloatTensor),
            )
            for path, label in self.files
        ]
        return self

    def to_cuda(self):
        cuda_images = None
        if torch.cuda.is_available():
            cuda_images = [(im.to("cuda"), c.to("cuda")) for im, c in self.images]
        self.images = cuda_images
        return self

    def free_memory(self):
        # destroy all images
        for im, c in self.images:
            im = im.to("cpu")
            c = c.to("cpu")
            del im
            del c

        # clear the list
        self.images.clear()

        # remove data from cuda
        torch.cuda.empty_cache()

        # force garbage collector
        gc.collect()
        return self

    def __getitem__(self, idx):
        # should return a image and a label
        return self.images[idx]

    def __len__(self):
        # should return the number of images on the dataset
        return len(self.files)

    def get_class_weights(self):
        targets = [c.item() for _, c in self.images]
        balance = np.mean(targets)
        return torch.FloatTensor([1 - balance, balance])
