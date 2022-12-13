from pydantic import BaseModel, PositiveFloat, PositiveInt, validator 
from racing_toolbox.observation.utils import ScreenFrame
from typing import NamedTuple

class VAEConfig(BaseModel):
    wandb_checkpoint_ref: str


class VAETrainingConfig(BaseModel):
    observation_frame: ScreenFrame
    lr: PositiveFloat
    epochs: PositiveInt
    kld_coeff: PositiveFloat
    latent_dim: PositiveInt
    input_shape: tuple[PositiveInt, PositiveInt]
    validation_fraction: PositiveFloat
    batch_size: PositiveInt

    @validator("input_shape")
    def square_shape(cls, val):
        assert isinstance(val, tuple) and len(val) == 2
        assert val[0] == val[1], "Currently VAE can handle only squared obs shape"
        return val 

class ConvFilter(NamedTuple):
    out_channels: PositiveInt
    kernel: tuple[PositiveInt, PositiveInt]
    stride: PositiveInt


class VAEModelConfig(BaseModel):
    conv_filters: list[ConvFilter] = []
