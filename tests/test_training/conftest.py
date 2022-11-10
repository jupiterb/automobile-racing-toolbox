import pytest
import gym
from gym import spaces
from collections import namedtuple
import random

from racing_toolbox.training.config import DQNConfig, ReplayBufferConfig, ModelConfig


SpaceParam = namedtuple("SpaceParam", ["observation", "action", "reward"])


class RandomEnv(gym.Env):
    def __init__(
        self, obs_space: spaces.Box, action_space: spaces.Space, reward_space: list
    ):
        self.observation_space = obs_space
        self.action_space = action_space
        self.reward_space = reward_space

    def step(self, *args, **kwargs):
        obs = self.observation_space.sample()
        rew = random.choice(self.reward_space)
        done = False
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
def fake_env(request) -> str:
    from ray.tune.registry import register_env

    spaces: SpaceParam = request.param
    obs_space, action_space, reward_space = spaces
    register_env(
        "my_test_env", lambda _: RandomEnv(obs_space, action_space, reward_space)
    )
    return "my_test_env"
