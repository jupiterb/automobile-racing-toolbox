from __future__ import annotations
import torch
import torch as th
from torch.nn import functional as F
from typing import Optional
import pytorch_lightning as pl
from torchvision.utils import make_grid
from dataclasses import dataclass, field, asdict
import wandb, os
from pathlib import Path

import racing_toolbox.observation.config.vae_config as configs

DEVICE = "cuda" if th.cuda.is_available() else "cpu"


def load_vae_from_wandb_checkpoint(checkpoint_location: str) -> VanillaVAE:
    finish_at_end = False
    run = wandb.run
    if run is None:
        run = wandb.init()
        finish_at_end = True
    artifact = run.use_artifact(checkpoint_location, type="model")
    artifact_dir = artifact.download()
    if finish_at_end:
        wandb.finish()
    model = VAE.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    model.vae.eval()
    return model.vae


def DownamplingBlock(in_channels, n_dims, stride=2) -> th.nn.Module:
    return th.nn.Sequential(
        th.nn.Conv2d(
            in_channels, out_channels=n_dims, kernel_size=3, stride=stride, padding=1
        ),
        th.nn.BatchNorm2d(n_dims),
        th.nn.LeakyReLU(),
    )


def UpsamplingBlock(in_channels, n_dims, stride=2) -> th.nn.Module:
    return th.nn.Sequential(
        th.nn.ConvTranspose2d(
            in_channels,
            n_dims,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=1,
        ),
        th.nn.BatchNorm2d(n_dims),
        th.nn.LeakyReLU(),
    )


class Encoder(th.nn.Module):
    def __init__(
        self,
        filters: list[configs.ConvFilter],
        in_channels: int,
        latent_dim: int,
        input_shape: tuple[int, int],
    ):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        modules: list[th.nn.Module] = []
        channels_in = in_channels
        for filter in filters:
            block = DownamplingBlock(
                in_channels=channels_in,
                n_dims=filter.out_channels,
                stride=filter.stride,
            )
            modules.append(block)
            channels_in = filter.out_channels

        self.encoder = th.nn.Sequential(*modules)
        with th.no_grad():
            out_sample = self.encoder(th.ones((1, in_channels, *input_shape)))
        self.before_flat_shape = out_sample.squeeze(0).shape

        self.encoder_fc = th.nn.Linear(
            out_sample.view(-1).size()[0], filters[-1].out_channels
        )

        self.fc_mu = th.nn.Linear(filters[-1].out_channels, latent_dim)
        self.fc_var = th.nn.Linear(filters[-1].out_channels, latent_dim)

    def reparameterize(self, mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * th.randn(mu.size()).to(DEVICE)

    def forward(self, X: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        X = self.encoder(X)
        X = torch.flatten(X, start_dim=1)

        X = self.encoder_fc(X)
        X = th.nn.functional.leaky_relu(X)

        mu = self.fc_mu(X)
        log_var = self.fc_var(X)
        return mu, log_var


class Decoder(th.nn.Module):
    def __init__(
        self,
        filters: list[configs.ConvFilter],
        out_channels: int,
        latent_dim: int,
        after_flat_shape: tuple[int, int, int],
    ):
        super(Decoder, self).__init__()
        self.c, self.w, self.h = after_flat_shape
        flat_size = self.c * self.w * self.h

        self.decoder_fc = th.nn.Sequential(
            th.nn.Linear(latent_dim, filters[0].out_channels),
            th.nn.LeakyReLU(),
            th.nn.Linear(filters[0].out_channels, flat_size),
            th.nn.LeakyReLU(),
        )
        modules: list[th.nn.Module] = []
        channels_in = self.c
        for filter in filters:
            block = UpsamplingBlock(
                in_channels=channels_in,
                n_dims=filter.out_channels,
                stride=filter.stride,
            )
            modules.append(block)
            channels_in = filter.out_channels
        last_block = th.nn.Conv2d(channels_in, out_channels, 1, 1)

        self.decoder = th.nn.Sequential(*modules, last_block)

    def forward(self, Z: th.Tensor) -> th.Tensor:
        X = self.decoder_fc(Z)
        X = X.view(-1, self.c, self.h, self.w)
        X = self.decoder(X)
        X = F.sigmoid(X)
        return X


class VanillaVAE(th.nn.Module):
    def __init__(
        self,
        filters: list[configs.ConvFilter],
        in_channels: int,
        latent_dim: int,
        input_shape: tuple[int, int] = (128, 128),
    ) -> None:
        super(VanillaVAE, self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        filters = [
            configs.ConvFilter(out_channels=t[0], kernel=t[1], stride=t[2])
            for t in filters
        ]
        self.encoder = Encoder(filters, in_channels, latent_dim, input_shape)
        self.decoder = Decoder(
            filters[::-1],
            in_channels,
            latent_dim,
            self.encoder.before_flat_shape,
        )

    def forward(
        self, X: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        mu, log_var = self.encoder(X)
        z = self.encoder.reparameterize(mu, log_var)
        return self.decoder(z), X, mu, log_var

    def loss_function(self, recons, input, mu, log_var, kld_weight: float) -> dict:
        recons_loss = F.mse_loss(recons, input)
        var = th.exp(log_var)
        kld_loss = 0.5 * th.mean(th.sum(1 + log_var - var - mu**2, dim=1), dim=0)

        loss = recons_loss - kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def sample(self, num_samples: int) -> th.Tensor:
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(DEVICE)

        samples = self.decoder(z)
        return samples

    def generate(self, x: th.Tensor) -> th.Tensor:
        return self.forward(x)[0]

    def to_latent(self, x: th.Tensor) -> th.Tensor:
        mu, _ = self.encoder(x.to(DEVICE))
        return mu


class VAE(pl.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.vae = VanillaVAE(
            params["conv_filters"],
            params["in_channels"],
            params["latent_dim"],
            params["input_shape"],
        ).to(
            DEVICE
        )  # TODO: add input shape
        self._val_batches = 0
        self.is_kld_anealing = {
            "kld_max_duration",
            "kld_max",
            "kld_aneal_duration",
        }.issubset(set(params.keys()))
        if self.is_kld_anealing:
            self.kld_scheduler = KLDScheduler(
                params["kld_coeff"],
                params["kld_max"],
                params["kld_max_duration"],
                params["kld_aneal_duration"],
            )

    def training_step(self, batch, batch_idx) -> dict[str, th.Tensor]:
        if batch_idx == 0 and self.is_kld_anealing:
            self.params["kld_coeff"] = self.kld_scheduler.get()

        X = batch[0]
        X_hat, X, mu, log_var = self.vae(X)
        info = self.vae.loss_function(
            X_hat, X, mu, log_var, kld_weight=self.params["kld_coeff"]
        )
        self.log_dict(info | {"kld_coeff": self.params["kld_coeff"]})
        return info["loss"]

    def validation_step(self, batch, batch_idx) -> dict[str, th.Tensor]:
        X = batch[0]
        X_hat, X, mu, log_var = self.vae(X)
        info = self.vae.loss_function(
            X_hat, X, mu, log_var, kld_weight=self.params["kld_coeff"]
        )
        self.log_dict({name + "_val": value for name, value in info.items()})

        if batch_idx == 0:
            self._val_batches += 1
            if self._val_batches % 10 == 0:
                print(self._val_batches)
                X_grid = make_grid(X, 8)
                X_hat_grid = make_grid(X_hat, 8)
                self.logger.log_image(
                    key="validation_batch",
                    images=[X_grid, X_hat_grid],
                    caption=["original", "reconstructed"],
                )
        return info["loss"]

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.params["lr"])
        return optimizer


class KLDScheduler:
    def __init__(self, start_val, max_val, duration_before_reset, aneal_duration):
        self.start_val = start_val
        self.max_val = max_val
        self.decay = (max_val - start_val) / aneal_duration
        self.duration_before_reset = duration_before_reset

        self.current_val = self.start_val
        self.steps_in_max = 0

    def get(self) -> float:
        if self.current_val >= self.max_val:
            if self.steps_in_max >= self.duration_before_reset:
                self.steps_in_max = 0
                self.current_val = self.start_val
            else:
                self.steps_in_max += 1
        else:
            self.current_val += self.decay
        return self.current_val


if __name__ == "__main__":
    params = configs.VAETrainingConfig(
        lr=0.1,
        epochs=10,
        kld_coeff=0.0001,
        latent_dim=8,
        input_shape=(128, 128),
        validation_fraction=0.1,
        batch_size=64,
        observation_frame=configs.ScreenFrame(),
    )
    filters = [
        (16, [3, 3], 2),
        (32, [3, 3], 2),
        (64, [3, 3], 2),
        (128, [3, 3], 2),
        (128, [3, 3], 2),
    ]
    model_config = configs.VAEModelConfig(conv_filters=filters)
    vae = VAE(params.dict() | model_config.dict() | {"in_channels": 1})
    i = th.ones((1, 1, 128, 128))

    print(vae.vae(i)[0].shape)

    print(params.dict() | model_config.dict() | {"in_channels": 3})
