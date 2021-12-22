import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

INPUT_DIM = 24
NUM_CHANNELS = 3
LATENT_VARIABLES = 2
HIDDEN_UNITS = 500

class GaussianMLPEncoder(pl.LightningModule):
    def __init__(self, input_size: int, hidden_units: int, latent_variables: int):
        super().__init__()
        self.W1 = nn.Linear(input_size, hidden_units, bias=True)
        self.W2 = nn.Linear(hidden_units, latent_variables, bias=True)
        self.W3 = nn.Linear(hidden_units, latent_variables, bias=True)

        nn.init.normal_(self.W1.weight, mean=0, std=0.01)
        nn.init.normal_(self.W2.weight, mean=0, std=0.01)
        nn.init.normal_(self.W3.weight, mean=0, std=0.01)

    def forward(self, x):
        # x has shape (batch_size, height*width)
        h = torch.relu(self.W1(x))  # paper uses tanh instead of relu
        mu = self.W2(h)
        logvar = self.W3(h)
        return (mu, logvar)


class GaussianMLPDecoder(pl.LightningModule):
    def __init__(self, input_size: int, hidden_units: int, latent_variables: int):
        super().__init__()
        self.W1 = nn.Linear(latent_variables, hidden_units, bias=True)
        self.W2 = nn.Linear(hidden_units, input_size, bias=True)
        self.W3 = nn.Linear(hidden_units, input_size, bias=True)

        nn.init.normal_(self.W1.weight, mean=0, std=0.01)
        nn.init.normal_(self.W2.weight, mean=0, std=0.01)
        nn.init.normal_(self.W3.weight, mean=0, std=0.01)

    def forward(self, z):
        # z has shape (batch_size, latent_variables)
        h = torch.relu(self.W1(z))  # paper uses tanh instead of relu
        mu = torch.sigmoid(self.W2(h))
        logvar = self.W3(h)
        return (mu, logvar)

    def compute_loss(self, x, mu_x, logvar_x):
        # return nn.functional.mse_loss(pred, inp, reduction='mean')
        weight = - 0.5 * logvar_x.exp()
        return (weight * (x - mu_x).pow(2)).mean()


class BernoulliMLPDecoder(pl.LightningModule):
    def __init__(self, input_size: int, hidden_units: int, latent_variables: int):
        super().__init__()
        self.W1 = nn.Linear(latent_variables, hidden_units, bias=True)
        self.W2 = nn.Linear(hidden_units, input_size, bias=True)

        nn.init.normal_(self.W1.weight, mean=0, std=0.01)
        nn.init.normal_(self.W2.weight, mean=0, std=0.01)

    def forward(self, z):
        # paper uses tanh instead of relu
        return torch.sigmoid(self.W2(torch.relu(self.W1(z))))

    def compute_loss(self, inp, pred):
        return nn.functional.binary_cross_entropy(pred, inp, reduction='sum')


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, input_dim=INPUT_DIM, num_channels=NUM_CHANNELS, hidden_units=HIDDEN_UNITS, latent_variables=LATENT_VARIABLES, decoder_distribution="bernoulli"):
        super().__init__()
        input_size = int(input_dim**2*num_channels)
        self.rearrange = Rearrange('b c h w -> b (c h w)')
        self.encoder = GaussianMLPEncoder(
            input_size, hidden_units, latent_variables)

        if decoder_distribution == "bernoulli":
            self.decoder = BernoulliMLPDecoder(
                input_size, hidden_units, latent_variables
            )
        elif decoder_distribution == "gaussian":
            self.decoder = GaussianMLPDecoder(
                input_size, hidden_units, latent_variables
            )
        self.decoder_distribution = decoder_distribution
        self.latent_variables = latent_variables

    def sample(self, mu_z, logvar_z):
        epsilon = torch.randn_like(mu_z)
        return epsilon*torch.exp(0.5*logvar_z) + mu_z

    def compute_loss(self, batch):
        x, _ = batch
        x = self.rearrange(x)

        mu_z, logvar_z = self.encoder(x)

        kl_loss = -0.5 * torch.sum(1 + logvar_z -
                                   mu_z.pow(2) - logvar_z.exp())

        z = self.sample(mu_z, logvar_z)

        if self.decoder_distribution == "bernoulli":
            mu_x = self.decoder(z)
            reconstruction_loss = self.decoder.compute_loss(inp=x, pred=mu_x)
        elif self.decoder_distribution == "gaussian":
            mu_x, logvar_x = self.decoder(z)
            reconstruction_loss = self.decoder.compute_loss(x, mu_x, logvar_x)

        return kl_loss + reconstruction_loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("training_loss", loss/len(batch[0]), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('test_loss', loss/len(batch[0]), on_step=False,
                 on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("valid_loss", loss/len(batch[0]), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # torch.optim.Adagrad(self.parameters(), lr=0.02, weight_decay=0.01)  # as in the paper
        return torch.optim.Adam(self.parameters(), lr=1e-3)
