import os

import numpy as np
import torch
from einops import rearrange
from scipy.stats import norm
from torchvision.transforms import ToPILImage

from data_utils import MNIST_utils, punks_utils
from model import VariationalAutoEncoder

CHECKPOINT_PATH = "logs/default/version_1/checkpoints/epoch=29-step=1799.ckpt"
SAVE_DIR = "./visualization_imgs"


def load_model(checkpoint_path=CHECKPOINT_PATH):
    return VariationalAutoEncoder.load_from_checkpoint(
        checkpoint_path=checkpoint_path)


def input_vs_reconstruction(num_samples=5, dataset="MNIST"):
    """
    Samples num_samples images from the test set and compare them side by side with their reconstruction.
    """
    vae = load_model()

    if dataset == "MNIST":
        dataloader = MNIST_utils.load_test_data(
            shuffle=True, batch_size=num_samples)
        height, width, channels = 28, 28, 1
    elif dataset == "punks":
        dataloader = punks_utils.load_test_data(
            shuffle=True, batch_size=num_samples)
        height, width, channels = 24, 24, 3

    batch = next(iter(dataloader))
    x, _ = batch

    x_flat = rearrange(x, "b c h w -> b (c h w)")
    mu_z, _ = vae.encoder(x_flat)
    mu_x = vae.decoder(mu_z)

    x_tilde = rearrange(mu_x, "b (c h w) -> c (b h) w",
                        h=height, w=width, c=channels)

    img = ToPILImage()(
        torch.cat([rearrange(x, "b c h w -> c (b h) w"), x_tilde], axis=-1)
    )
    img.save(os.path.join(SAVE_DIR, "%s_reconstruction.png" % dataset))


def plot_learned_manifold(n=10, dataset="punks"):
    # only applicable to 2D latent spaces
    # n represents the number of intervals

    if dataset == "punks":
        height, width, channels = 24, 24, 3
    elif dataset == "MNIST":
        height, width, channels = 28, 28, 1

    vae = load_model()

    # evenly divide the standard normal distribution by quantile
    grid = norm.ppf(np.linspace(0.05, 0.95, n))

    x = np.repeat(grid[:, np.newaxis], n, 1)
    y = np.repeat(grid[np.newaxis, :], n, 0)

    mu_z = torch.tensor(np.concatenate(
        [x[:, :, np.newaxis], y[:, :, np.newaxis]], axis=-1)).type(torch.FloatTensor)
    mu_z = mu_z.view(-1, 2)
    imgs = vae.decoder(mu_z)
    imgs = rearrange(imgs, "(n1 n2) (c h w) -> c (n1 h) (n2 w)",
                     h=height, w=width, c=channels, n1=n, n2=n)
    imgs = ToPILImage()(imgs)
    imgs.save(os.path.join(SAVE_DIR, "%s_learned_manifold.png" % dataset))


if __name__ == "__main__":
    input_vs_reconstruction(num_samples=5, dataset="punks")
    plot_learned_manifold(n=20, dataset="punks")
