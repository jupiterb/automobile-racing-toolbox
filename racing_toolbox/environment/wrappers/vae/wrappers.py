from racing_toolbox.environment.wrappers.vae import load_vae_from_wandb_checkpoint, VanillaVAE
import gym 
from gym import spaces 
import numpy as np 
import torch as th 
from  torchvision.transforms.functional import to_tensor


class VaeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, vae: VanillaVAE) -> None:
        super().__init__(env)
        self.vae = vae
        self.vae.eval()
        self.observation_space = spaces.Box(0, 1, (self.vae.latent_dim, ))
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """given RGB compatible with vae model input, return sample from latent space"""
        img = to_tensor(observation) / 255

        with th.no_grad():
            mu, log_var = self.vae(img.unsqueeze(0))
            latent_vec = self.vae.reparameterize(mu, log_var)
        return latent_vec.detach().squeeze(0).numpy()
