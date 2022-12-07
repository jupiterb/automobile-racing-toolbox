import pytest
from ray.rllib import algorithms as alg
from ray.rllib.algorithms import dqn, bc

from racing_toolbox.training.algorithm_constructor import construct_cls
from racing_toolbox.training.config import TrainingParams


@pytest.mark.parametrize("construct_training_config", [bc.BC, dqn.DQN], indirect=True)
def test_right_calsses_got_initialized_from_config(
    construct_training_config: tuple[TrainingParams, type[alg.Algorithm]]
):
    training_params, expected_algo_class = construct_training_config
    cls = construct_cls(training_params)
    assert isinstance(cls, expected_algo_class), "incorect class was constructed"
