import pytest
import gym
from gym import spaces
from collections import namedtuple
import random

from racing_toolbox.training.config import DQNConfig, ReplayBufferConfig, ModelConfig
from racing_toolbox.training.config.params import TrainingParams


SpaceParam = namedtuple("SpaceParam", ["observation", "action", "reward"])


class RandomEnv(gym.Env):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Space,
        reward_space: list,
        episode_len: int,
        name: str,
    ):
        self.observation_space = obs_space
        self.action_space = action_space
        self.reward_space = reward_space

        self.max_episode_len = episode_len
        self._curr_episode_len = 0
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def step(self, *args, **kwargs):
        self._curr_episode_len += 1
        obs = self.observation_space.sample()
        rew = random.choice(self.reward_space)
        done = False
        if self._curr_episode_len >= self.max_episode_len:
            self._curr_episode_len = 0
            done = True
        info = {}
        return obs, rew, done, info

    def reset(self):
        return self.observation_space.sample()

    def render(self, mode="human"):
        return self.observation_space.sample()


@pytest.fixture
def dqn_config():
    return DQNConfig(replay_buffer_config=ReplayBufferConfig())


@pytest.fixture
def model_config():
    return ModelConfig(fcnet_activation="relu", fcnet_hiddens=[100, 100])


@pytest.fixture
def fake_env(request) -> RandomEnv:
    from ray.tune.registry import register_env

    spaces: SpaceParam = request.param[0]
    max_ep_len = request.param[1]
    obs_space, action_space, reward_space = spaces
    register_env(
        "my_test_env",
        lambda _: RandomEnv(
            obs_space, action_space, reward_space, max_ep_len, "my_test_env"
        ),
    )
    return RandomEnv(obs_space, action_space, reward_space, max_ep_len, "my_test_env")
