import os

from pytorch_lightning import Trainer

from data_utils import MNIST_utils, punks_utils
from model import VariationalAutoEncoder
from visualizations import CHECKPOINT_PATH

CHECKPOINT_PATH = "./logs/vae_model/version_3/checkpoints/epoch=50-step=24479.ckpt"


def main(dataset="MNIST"):
    if dataset == "MNIST":
        test_dataloader = MNIST_utils.load_test_data()
        params = {
            "input_dim": 28,
            "num_channels": 28,
            "hidden_units": 500,
            "latent_variables": 2,
        }
    elif dataset == "punks":
        test_dataloader = punks_utils.load_test_data()
        params = {
            "input_dim": 24,
            "num_channels": 3,
            "hidden_units": 500,
            "latent_variables": 2,
        }

    vae = VariationalAutoEncoder.load_from_checkpoint(CHECKPOINT_PATH)

    trainer = Trainer()
    trainer.test(model=vae, dataloaders=test_dataloader)


if __name__ == '__main__':
    main(dataset="punks")
