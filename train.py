from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from data_utils import MNIST_utils, punks_utils
from model import VariationalAutoEncoder


def main(dataset="MNIST", checkpoint_path=None):

    if dataset == "punks":
        train_dataloader, valid_dataloader = punks_utils.load_train_and_valid_data()
        params = {
            'input_dim': 24,
            'num_channels': 3,
            'hidden_units': 500,
            'latent_variables': 2
        }
    elif dataset == "MNIST":
        train_dataloader, valid_dataloader = MNIST_utils.load_train_and_valid_data()
        params = {
            'input_dim': 28,
            'num_channels': 1,
            'hidden_units': 500,
            'latent_variables': 20
        }

    if checkpoint_path:
        vae = VariationalAutoEncoder.load_from_checkpoint(checkpoint_path)
    else:
        vae = VariationalAutoEncoder(**params)
    logger = TensorBoardLogger(save_dir="./logs")
    trainer = Trainer(
        logger=logger, max_epochs=30)
    trainer.fit(vae, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main(dataset="punks",
         checkpoint_path=None)
