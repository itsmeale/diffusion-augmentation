from pathlib import Path
from time import time
from uuid import uuid1

import torch
import torchvision as tvis
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet


class XRAYDiffusionModel:
    def __init__(self, images_path: str):
        self.model = Unet(
            dim=16,
            dim_mults=(1, 2, 4),
            channels=1,
        ).cuda()

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size=64,
            timesteps=150,
        ).cuda()

        self.trainer = Trainer(
            self.diffusion,
            images_path,
            train_batch_size=128,
            train_lr=1e-5,
            train_num_steps=50_000,
            gradient_accumulate_every=1,
            ema_decay=0.995,
            amp=False,
        )

    def fit(self):
        self.trainer.train()
        return self

    def sample(self, num_images: int):
        return self.diffusion.sample(batch_size=num_images)

    def save(self):
        torch.save(self.diffusion.state_dict(), "results/xray_model")
        return self

    def load(self):
        self.diffusion.load_state_dict(torch.load("results/xray_model"))
        return self


def save_imgs(imgs, folder):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)

    for img in imgs:
        t = uuid1()
        file_path = path / f"{t}.jpeg"
        tvis.utils.save_image(img, file_path)


def synthetize_train(ds_class):
    xray_diff = XRAYDiffusionModel(
        images_path=f"data/preprocessed/xray_resized/train/{ds_class}"
    )

    xray_diff.fit().save()

    model = xray_diff.load()

    images_to_sample = 3875
    sampled = 0
    batch = 500

    while sampled <= images_to_sample:
        images = model.sample(min(batch, images_to_sample - sampled))
        save_imgs(images, f"data/preprocessed/xray_generated/train/{ds_class}")
        sampled += batch


def main():
    classes = [
        "NORMAL",
        "PNEUMONIA",
    ]

    for c in classes:
        synthetize_train(c)


if __name__ == "__main__":
    main()
