# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Tuple

from loguru import logger
from PIL import Image
from tqdm import tqdm


class DatasetDownsize:
    def __init__(
        self,
        images_folder: str,
        destination_folder: str,
        desired_resolution: Tuple[int, int],
        extension: str,
    ):
        self.images_path = Path(images_folder)
        self.destination_folder = Path(destination_folder)
        self.desired_resolution = desired_resolution
        self.extension = extension

    def downsize(self):
        logger.info("Downsizing images")
        images_path = self.images_path.glob(f"*.{self.extension}")

        self.destination_folder.mkdir(parents=True, exist_ok=True)

        for im_path in tqdm(images_path):
            img = Image.open(im_path).resize(size=self.desired_resolution)
            img.save(self.destination_folder / im_path.name)

        return images_path


if __name__ == "__main__":
    d = DatasetDownsize(
        "data/raw/xray/train/NORMAL",
        "data/preprocessed/xray_resized/train/NORMAL",
        (64, 64),
        "jpeg",
    )

    d.downsize()
