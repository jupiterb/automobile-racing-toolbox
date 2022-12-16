import gym
import numpy as np
import wandb
import gym.spaces
import torch as th 
from torchvision import transforms 

from racing_toolbox.observation.config import LidarConfig, TrackSegmentationConfig
from racing_toolbox.observation.lidar import Lidar
from racing_toolbox.observation.track_segmentation import TrackSegmenter
from racing_toolbox.observation.utils import ScreenFrame
from racing_toolbox.environment.utils.logging import log_observation
from racing_toolbox.observation.vae import VanillaVAE
from concurrent.futures import ThreadPoolExecutor


class VaeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, vae: VanillaVAE) -> None:
        super().__init__(env)
        self.vae = vae
        self.vae.eval()
        self.observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self.vae.latent_dim, )
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(vae.input_shape)
        ])

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
            np.max(env.observation_space.high), self.move_axis(env.observation_space.sample()).shape
        )

    @log_observation(__name__)
    def observation(self, observation: np.ndarray):
        observation = np.squeeze(observation)
        return self.move_axis(observation)


class RescaleWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, env.observation_space.sample().shape)

    @log_observation(__name__)
    def observation(self, observation: np.ndarray):
        return observation / 255.0


class CutImageWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, frame: ScreenFrame) -> None:
        super().__init__(env)
        self._frame = frame 
        shape = frame.apply(env.observation_space.sample()).shape
        self.observation_space = gym.spaces.Box(
            np.min(env.observation_space.low),
            np.max(env.observation_space.high),
            shape
        )

    @log_observation(__name__)
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self._frame.apply(observation)


class LidarWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: LidarConfig) -> None:
        super().__init__(env)
        self._lidar = Lidar(config)
        self.observation_space = gym.spaces.Box(
            0, 1, (len(range(*config.angles_range)) + 1, config.depth)
        )

    @log_observation(__name__)
    def observation(self, observation: np.ndarray):
        return self._lidar.scan_2d(observation)[0]


class TrackSegmentationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: TrackSegmentationConfig) -> None:
        super().__init__(env)
        self._track_segmenter = TrackSegmenter(config)
        # TODO(jupiterb): add observation space 

    @log_observation(__name__)
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self._track_segmenter.perform_segmentation(observation)

def log_video(imgs: list[np.ndarray]):
    wandb.run.log({"recording": wandb.Video(np.stack(imgs), fps=10)})

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
        img = np.moveaxis(obs, -1, 0) # channel first
        self._frames.append(img)
        if self.log_duration == len(self._frames):
            print("logging video")
            self.pool.submit(log_video, self._frames)
            print("logged video")
