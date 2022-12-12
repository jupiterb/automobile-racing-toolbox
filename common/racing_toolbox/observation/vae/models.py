from __future__ import annotations
import torch
import torch as th 
from torch.nn import functional as F
from typing import Optional 
import pytorch_lightning as pl
from torchvision.utils import make_grid
from dataclasses import dataclass, field
import wandb, os
from pathlib import Path 

DEVICE = "cuda" if th.cuda.is_available() else "cpu"


def load_vae_from_wandb_checkpoint(checkpoint_location: str) -> VanillaVAE:
    run = wandb.run or wandb.init()
    artifact = run.use_artifact(checkpoint_location, type="model")
    artifact_dir = artifact.download()
    model = VAE.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    model.vae.eval()
    return model.vae 


def DownamplingBlock(in_channels, n_dims) -> th.nn.Module:
    return th.nn.Sequential(
        th.nn.Conv2d(
            in_channels,
            out_channels=n_dims,
            kernel_size=3,
            stride=2,
            padding=1
        ),
        th.nn.BatchNorm2d(n_dims),
        th.nn.LeakyReLU(),
    )

def UpsamplingBlock(in_channels, n_dims) -> th.nn.Module:
    return th.nn.Sequential(
        th.nn.ConvTranspose2d(
            in_channels,
            n_dims,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        ),
        th.nn.BatchNorm2d(n_dims),
        th.nn.LeakyReLU(),
    )

class VanillaVAE(th.nn.Module):
    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: Optional[list[int]] = None, input_shape: tuple[int, int] = (128, 128)
    ) -> None:
        super(VanillaVAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.h, self.w = 4, 4
        self.hidden_dims = hidden_dims or [16, 32, 64, 128, 128]

        modules: list[th.nn.Module] = []
        encoder_dims = [in_channels] + self.hidden_dims
        for i in range(len(encoder_dims) - 1):
            modules.append(
                DownamplingBlock(encoder_dims[i], encoder_dims[i + 1])
            )
        self.encoder = th.nn.Sequential(*modules)
        self.encoder_fc = th.nn.Linear(self.hidden_dims[-1]  * self.h * self.w, self.hidden_dims[-1])

        self.fc_mu = th.nn.Linear(self.hidden_dims[-1], latent_dim)
        self.fc_var = th.nn.Linear(self.hidden_dims[-1], latent_dim)

        decoder_dims = self.hidden_dims[::-1] + [in_channels]
        self.decoder_fc = th.nn.Sequential(
            th.nn.Linear(latent_dim, self.hidden_dims[-1]),
            th.nn.LeakyReLU(),
            th.nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] * self.w * self.h),
            th.nn.LeakyReLU()
        )
        modules: list[th.nn.Module] = []
        for i in range(len(decoder_dims) - 1):
            modules.append(
                UpsamplingBlock(decoder_dims[i], decoder_dims[i + 1])
            )

        self.decoder = th.nn.Sequential(*modules)


    def encode(self, X: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        X = self.encoder(X)
        X = torch.flatten(X, start_dim=1)

        X = self.encoder_fc(X)
        X = th.nn.functional.leaky_relu(X)

        mu = self.fc_mu(X)
        log_var = self.fc_var(X)
        return mu, log_var

    def decode(self, Z: th.Tensor) -> th.Tensor:
        X = self.decoder_fc(Z)
        X = X.view(-1, self.hidden_dims[-1], self.h, self.w)
        X = self.decoder(X)
        X = F.sigmoid(X)
        return X

    def reparameterize(self, mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * th.randn(mu.size()).to(DEVICE)

    def forward(self, X: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        mu, log_var = self.encode(X)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), X, mu, log_var

    def loss_function(self, recons, input, mu, log_var, kld_weight: float) -> dict:
        recons_loss = F.mse_loss(recons, input)
        var = th.exp(log_var)
        kld_loss = 0.5 * th.mean(
            th.sum(1 + log_var - var - mu ** 2, dim=1), 
            dim=0 
        )

        loss = recons_loss - kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def sample(self, num_samples: int) -> th.Tensor:
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(DEVICE)

        samples = self.decode(z)
        return samples

    def generate(self, x: th.Tensor) -> th.Tensor:
        return self.forward(x)[0]

    def to_latent(self, x: th.Tensor) -> th.Tensor:
        mu, log_var = self.encode(x.to(DEVICE))
        latent_vec = self.reparameterize(mu, log_var)
        return latent_vec


@dataclass 
class TrainingParams:
    lr: float
    epochs: int 
    kld_coeff: float
    latent_dim: int 
    input_shape: tuple[int, int]
    validation_fraction: float = 0.1
    batch_size: int = 64
    hiddens : list[int] = field(default_factory=lambda: list([8, 16, 32, 32, 32]))


class VAE(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.vae = VanillaVAE(3, params["latent_dim"], params["hiddens"]).to(DEVICE) # TODO: add input shape 

    def training_step(self, batch, batch_idx) -> dict[str, th.Tensor]:
        X = batch[0]
        X_hat, X, mu, log_var = self.vae(X)
        info = self.vae.loss_function(X_hat, X, mu, log_var, kld_weight=self.params["kld_coeff"])
        self.log_dict(info)
        return info["loss"]

    def validation_step(self, batch, batch_idx) -> dict[str, th.Tensor]:
        X = batch[0]
        X_hat, X, mu, log_var = self.vae(X)
        info = self.vae.loss_function(X_hat, X, mu, log_var, kld_weight=self.params["kld_coeff"])
        self.log_dict({name + "_val": value for name, value in info.items()})
        if batch_idx == 0:
            X_grid = make_grid(X, 8)
            X_hat_grid = make_grid(X_hat, 8)
            self.logger.log_image(key="validation_batch", images=[X_grid, X_hat_grid], caption=["original", "reconstructed"])
        return info["loss"]

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.params["lr"])
        return optimizer