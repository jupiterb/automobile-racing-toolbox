import pytest
from racing_toolbox.trainer import Trainer
from racing_toolbox.trainer.algorithm_constructor import construct_cls
from racing_toolbox.trainer.config import TrainingConfig, TrainingParams
from tests.test_training.conftest import SpaceParam
from gym.spaces import Box, Discrete


@pytest.mark.parametrize(
    "fake_env",
    [
        SpaceParam(Box(-1, 1, (100, 100)), Discrete(5), [1]),  # GRAYSCALE
        SpaceParam(Box(-1, 1, (100, 100, 3)), Discrete(5), [1]),  # RGB
        SpaceParam(
            Box(-1, 1, (4, 100, 100)), Discrete(5), [1]
        ),  # framestack + grayscale
    ],
    indirect=True,
)
def test_if_convolution_model_fit_to_obs_space(fake_env, model_config, dqn_config):
    raise NotImplementedError
