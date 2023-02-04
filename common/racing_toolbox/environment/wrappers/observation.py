import gym
import numpy as np
import wandb
import gym.spaces
import torch as th
from torchvision import transforms
import logging

from racing_toolbox.observation.config import LidarConfig, TrackSegmentationConfig
from racing_toolbox.observation.lidar import Lidar
from racing_toolbox.observation.track_segmentation import TrackSegmenter
from racing_toolbox.observation.utils import ScreenFrame
from racing_toolbox.environment.utils.logging import log_observation
from racing_toolbox.observation.vae import VanillaVAE
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class SpeedAppendingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale: int):
        super().__init__(env)
        self.scale = scale 
        assert len(env.observation_space.shape) == 1, f"{type(self)} works only on vector spaces"
        low = min(self.observation_space.low.min(), 0)
        high = max(self.observation_space.high.max(), 1) 
        size = self.observation_space.shape[0] + 1
        self.observation_space = gym.spaces.Box(low, high, (size, ))

    def step(self, action):
        obs, rew, done, info = super().step(action)
        new_obs = np.concatenate([obs, np.array([info["speed"] / self.scale])], dtype=np.float32)
        return new_obs, rew, done, info 

    def reset(self):
        obs = super().reset()
        new_obs = np.concatenate([obs, np.array([0])], dtype=np.float32)
        return new_obs


class VaeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, vae: VanillaVAE) -> None:
        super().__init__(env)
        self.vae = vae
        self.vae.eval()
        self.observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self.vae.latent_dim,),
        )
        transform_list = [transforms.ToTensor(), transforms.Resize(vae.input_shape)]
        if vae.in_channels == 1:
            transform_list.append(transforms.Grayscale())
        self.transform = transforms.Compose(
            transform_list
        )  # TODO: transforms stuff should be passed in constructor, or be implemented in vae class to avoid duplication

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """given RGB compatible with vae model input, return sample from latent space"""
        img: th.Tensor = self.transform(observation)

        with th.no_grad():
            latent_vec = self.vae.to_latent(img.unsqueeze(0))
        return latent_vec.detach().squeeze(0).numpy()


class SqueezingWrapper(gym.ObservationWrapper):
    """This wrapper applies np.squeeze to make shape of observation compatible with stb3"""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.move_axis = lambda obs: np.moveaxis(obs, 0, -1)
        self.observation_space = gym.spaces.Box(
            np.min(env.observation_space.low),
            np.max(env.observation_space.high),
            self.move_axis(env.observation_space.sample()).shape,
        )

    @log_observation(__name__)
    def observation(self, observation: np.ndarray):
        observation = np.squeeze(observation)
        return self.move_axis(observation)


class RescaleWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            0, 1, env.observation_space.sample().shape
        )

    @log_observation(__name__)
    def observation(self, observation: np.ndarray):
        return observation / 255.0


class CutImageWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, frame: ScreenFrame) -> None:
        super().__init__(env)
        self._frame = frame
        shape = frame.apply(env.observation_space.sample()).shape
        self.observation_space = gym.spaces.Box(
            np.min(env.observation_space.low), np.max(env.observation_space.high), shape
        )

    @log_observation(__name__)
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self._frame.apply(observation)


class LidarWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: LidarConfig) -> None:
        super().__init__(env)
        self._lidar = Lidar(config)
        self.observation_space = gym.spaces.Box(
            0, 1, ((len(range(*config.angles_range)) + 1) * config.depth,)
        )

    @log_observation(__name__)
    def observation(self, observation: np.ndarray):
        return self._lidar.scan_2d(observation)[0].flatten()


class TrackSegmentationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: TrackSegmentationConfig) -> None:
        super().__init__(env)
        self._track_segmenter = TrackSegmenter(config)
        # TODO(jupiterb): add observation space

    @log_observation(__name__)
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self._track_segmenter.perform_segmentation(observation)


def log_video(imgs: list[np.ndarray], key_name: str = "recording"):
    wandb.run.log({key_name: wandb.Video(np.stack(imgs), fps=10)})


class WandbVideoLogger(gym.Wrapper):
    def __init__(self, env: gym.Env, log_frequency: int, log_duration: int) -> None:
        super().__init__(env)
        assert log_frequency > log_duration

        self.log_freq = log_frequency
        self.log_duration = log_duration

        self._is_recording: bool = True
        self._frames: list[np.ndarray] = []
        self._step = 0

        self.pool = ThreadPoolExecutor(max_workers=10)

    def step(self, action):
        obs, rew, done, info = super().step(action)
        if wandb.run is not None:
            self._maybe_record(obs)
        return obs, rew, done, info

    def _maybe_record(self, obs: np.ndarray):
        if self._step % self.log_freq == 0 or self._is_recording:
            self._is_recording = True
            self._record(obs)
        if len(self._frames) == self.log_duration:
            self._is_recording = False
            self._frames = []

    def _record(self, obs: np.ndarray):
        img = np.moveaxis(obs, -1, 0)  # channel first
        self._frames.append(img)
        if self.log_duration == len(self._frames):
            logger.info("logging video")
            self.pool.submit(log_video, self._frames)
            logger.info("logged video")


class VaeVideoLogger(WandbVideoLogger):
    def __init__(
        self,
        env: gym.Env,
        log_frequency: int,
        log_duration: int,
        vae: VanillaVAE,
        decode_only: bool = False,
    ) -> None:
        """if decode_only is set, will assume that given observation is latent vector, and use only Decoder to log frame"""
        super().__init__(env, log_duration=log_duration, log_frequency=log_frequency)
        self.vae = vae
        self.vae.eval()
        transform_list = [transforms.ToTensor(), transforms.Resize(vae.input_shape)]
        if vae.in_channels == 1:
            transform_list.append(transforms.Grayscale())
        self.transform = transforms.Compose(transform_list)
        self.decode_only = decode_only

    def _record(self, obs: np.ndarray):
        if self.decode_only:
            with th.no_grad():
                obs_torch = th.Tensor(obs).unsqueeze(0)
                decoded = self.vae.decoder(obs_torch)
        else:
            obs_torch: th.Tensor = self.transform(obs)
            with th.no_grad():
                decoded = self.vae.generate(obs_torch.unsqueeze(0))
        img = decoded.detach().squeeze(0).numpy() * 255
        img = img.astype(np.uint8)

        self._frames.append(img)
        if self.log_duration == len(self._frames):
            logger.info("logging VAE video")
            self.pool.submit(log_video, self._frames, key_name="vae_reconstruction")
            logger.info("logged VAE video")
