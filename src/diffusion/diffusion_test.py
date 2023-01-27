from pathlib import Path

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
            self.model, image_size=64, timesteps=1, loss_type="l1"
        ).cuda()

        self.trainer = Trainer(
            self.diffusion,
            images_path,
            train_batch_size=128,
            train_lr=1e-5,
            train_num_steps=20_000,
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
        self.diffusion = GaussianDiffusion(
            self.model, image_size=64, timesteps=1, loss_type="l1"
        ).cuda()
        self.diffusion.load_state_dict(torch.load("results/xray_model"))
        return self


def save_imgs(imgs, folder):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(imgs):
        file_path = path / f"{i}.jpeg"
        tvis.utils.save_image(img, file_path)


def main():
    xray_diff = XRAYDiffusionModel(
        images_path="data/preprocessed/xray_resized/train/NORMAL"
    )
    images = xray_diff.fit().save().load().sample(2534)
    save_imgs(images, "data/preprocessed/xray_generated/train/NORMAL")


if __name__ == "__main__":
    main()
