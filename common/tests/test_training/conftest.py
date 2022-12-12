import pytest
import gym
from gym import spaces
from collections import namedtuple
import random
from ray.rllib.algorithms import dqn
from ray.rllib.algorithms import bc
from pathlib import Path
from gym.spaces import Box, Discrete
from dotenv import load_dotenv
from tests import TEST_DIR

from racing_toolbox.observation.utils.ocr.abstract import OcrTool
from racing_toolbox.observation.utils.ocr.seven_segments import SevenSegmentsOcr
from racing_toolbox.training.config import DQNConfig, ReplayBufferConfig, ModelConfig
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.training.config import (
    TrainingConfig,
    BCConfig,
    EvalConfig,
)
from racing_toolbox.datatool import DatasetContainer
from racing_toolbox.datatool.datasets import FromMemoryDataset
from racing_toolbox.datatool.preproc.rllib_ds import make_rllib_dataset
from racing_toolbox.datatool.utils.dataset_based_env import DatasetBasedEnv

load_dotenv(TEST_DIR / "assets/.env")

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

    try:
        spaces: SpaceParam = request.param[0]
        max_ep_len = request.param[1]
    except (IndexError, AttributeError):
        spaces, max_ep_len = (SpaceParam(Box(-1, 1, (84, 84, 4)), Discrete(5), [1]), 5)

    obs_space, action_space, reward_space = spaces
    register_env(
        "my_test_env",
        lambda _: RandomEnv(
            obs_space, action_space, reward_space, max_ep_len, "my_test_env"
        ),
    )
    return RandomEnv(obs_space, action_space, reward_space, max_ep_len, "my_test_env")


@pytest.fixture
def offline_data_path(game_conf, env_config, tmp_path):
    game = "trackmania"
    user = "test"
    name = "small"

    container = DatasetContainer()
    dataset = FromMemoryDataset(f"{TEST_DIR}/assets/recordings", game, user, name)
    assert container.try_add(dataset)

    ocr_tool = OcrTool(game_conf.ocrs, SevenSegmentsOcr)
    dataset_env = DatasetBasedEnv(container, ocr_tool)
    offline_dataset_path, _ = make_rllib_dataset(
        dataset_env, env_config, tmp_path, game, user, name
    )
    return offline_dataset_path


@pytest.fixture
def construct_training_config(
    request,
    fake_env,
    eval_conf: EvalConfig,
    training_config: TrainingConfig,
    offline_data_path: Path,
):
    params = TrainingParams(
        **training_config.dict(),
        input_=None,
        env_name=fake_env.name,
        observation_space=fake_env.observation_space,
        action_space=fake_env.action_space,
    )
    if request.param == bc.BC:
        params.algorithm = BCConfig()
        params.offline_data = [offline_data_path]
        params.evaluation_config = eval_conf
        return params, bc.BC
    if request.param == dqn.DQN:
        params.offline_data = None
        params.evaluation_config = None
        return params, dqn.DQN


@pytest.fixture
def bc_training_params(training_config, fake_env, eval_conf, offline_data_path):
    training_config.algorithm = BCConfig()
    training_config.evaluation_config = eval_conf
    params = TrainingParams(
        **training_config.dict(),
        env_name=fake_env.name,
        observation_space=fake_env.observation_space,
        action_space=fake_env.action_space,
        input_=[offline_data_path],
    )
    return params
