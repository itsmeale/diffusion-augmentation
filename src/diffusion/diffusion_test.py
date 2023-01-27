import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet
from torchvision.utils import make_grid

model = Unet(
    dim=16,
    dim_mults=(1, 2, 4),
    channels=1,
).cuda()

diffusion = GaussianDiffusion(
    model, image_size=64, timesteps=100, loss_type="l1"
).cuda()


trainer = Trainer(
    diffusion,
    "data/preprocessed/xray_resized/train/NORMAL",
    train_batch_size=128,
    train_lr=1e-5,
    train_num_steps=10_000,
    gradient_accumulate_every=1,
    ema_decay=0.995,
    amp=False,
)

trainer.train()

sampled_images = diffusion.sample(batch_size=10)

fig, ax = plt.subplots(dpi=300)
plt.imshow(make_grid(sampled_images.cpu(), 5).permute((1, 2, 0)))
plt.savefig("results_old.png")
