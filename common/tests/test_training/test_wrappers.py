import pytest
from racing_toolbox.observation.vae import load_vae_from_wandb_checkpoint 
from racing_toolbox.environment.wrappers import observation
import torch as th 
from tests.test_training.conftest import RandomEnv
import gym.spaces 
import numpy as np 


@pytest.fixture 
def vae_latent16_path():
    return "automobile-racing-toolbox/ART/model-3gdtl21q:best"



def test_model_loaded_properly(vae_latent16_path):
    model = load_vae_from_wandb_checkpoint(vae_latent16_path)
    model.eval()

    assert model.latent_dim == 16, "Invalid latent size"

    sample = th.zeros((1, 1, 128, 128))
    with th.no_grad():
        latent = model.to_latent(sample).detach().numpy()
    assert latent.shape == (1, 16), "Invalid latent vector shape"

def test_speed_append_wrapper(vae_latent16_path):
    env = RandomEnv(
        obs_space=gym.spaces.Box(0, 1, (128, 128, 3), dtype=np.float32),
        action_space=gym.spaces.Discrete(2),
        reward_space=[0, 1],
        episode_len=2,
        name="whatever"
    )
    model = load_vae_from_wandb_checkpoint(vae_latent16_path)
    model.eval()
    env = observation.VaeObservationWrapper(env, model)
    env = observation.SpeedAppendingWrapper(env, 300)
    assert env.step(0)[0].shape == (17, ), "wrong obs space"
    assert env.step(0)[0].dtype == np.float32