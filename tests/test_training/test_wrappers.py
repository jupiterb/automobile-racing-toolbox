import pytest
from racing_toolbox.environment.wrappers.vae import load_vae_from_wandb_checkpoint
import torch as th


@pytest.fixture
def vae_latent8_path():
    return "automobile-racing-toolbox/VAEv2/model-1g5v58m2:v149"


def test_model_loaded_properly(vae_latent8_path):
    model = load_vae_from_wandb_checkpoint(vae_latent8_path)
    model.eval()

    assert model.latent_dim == 8, "Invalid latent size"

    sample = th.zeros((1, 3, 128, 128))
    with th.no_grad():
        latent = model.to_latent(sample).detach().squeeze(0).numpy()
    assert latent.shape == (8,), "Invalid latent vector shape"
