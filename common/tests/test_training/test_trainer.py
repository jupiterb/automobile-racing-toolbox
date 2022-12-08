import pytest
from functools import reduce
from ray.rllib.algorithms import dqn, sac
from gym.spaces import Box, Discrete
from copy import deepcopy
from racing_toolbox.training import Trainer
from racing_toolbox.training.config.params import TrainingParams
from tests.test_training.conftest import SpaceParam


def training_test(training_params: TrainingParams):
    training_params.num_rollout_workers = 0
    # SAC needs more iterations, and it depends from observation space shape
    training_params.max_iterations = (
        reduce(lambda a, b: a * b, training_params.observation_space.shape) // 1_200
    )
    training_params.train_batch_size = 1
    training_params.checkpoint_frequency = 1000

    trainer = Trainer(training_params)
    weights_before = deepcopy(trainer.algorithm.get_policy().get_weights())
    trainer.run()

    assert str(weights_before) != str(
        trainer.algorithm.get_policy().get_weights()
    ), "weights are not changing"


@pytest.mark.parametrize(
    "fake_env",
    [
        (SpaceParam(Box(-1, 1, (84, 84)), Discrete(5), [1]), 5),
        (
            SpaceParam(Box(-1, 1, (84, 84, 4)), Discrete(5), [1]),
            5,
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("construct_training_config", [dqn.DQN, sac.SAC], indirect=True)
def test_if_training_runs_properly_with_discrete_action_space(
    construct_training_config,
):
    training_params, _ = construct_training_config
    training_test(training_params)

<<<<<<< HEAD:tests/test_training/test_trainer.py
=======
    trainer = Trainer(training_params)
    weights_before = deepcopy(trainer.algorithm.get_policy().get_weights())
    for _ in trainer.run():
        pass
>>>>>>> 2a43dfdba2e67dea4ff8ec1fd72a61921726cc4f:common/tests/test_training/test_trainer.py

@pytest.mark.parametrize(
    "fake_env",
    [
        (SpaceParam(Box(-1, 1, (84, 84)), Box(-1, 1, (3,)), [1]), 5),
    ],
    indirect=True,
)
@pytest.mark.parametrize("construct_training_config", [sac.SAC], indirect=True)
def test_if_training_runs_properly_with_continous_action_space(
    construct_training_config,
):
    training_params, _ = construct_training_config
    training_test(training_params)


@pytest.mark.parametrize(
    "fake_env",
    [
        (SpaceParam(Box(-1, 1, (84, 84)), Discrete(5), [1]), 5),
    ],
    indirect=True,
)
@pytest.mark.parametrize("construct_training_config", [dqn.DQN, sac.SAC], indirect=True)
def test_if_checkpoint_laoded(construct_training_config, tmp_path):
    training_params, _ = construct_training_config
    training_params.num_rollout_workers = 0
    training_params.max_iterations = 1
    training_params.train_batch_size = 1
    training_params.checkpoint_frequency = 1

    tmp_file = tmp_path / "checkpoint_dir/foo"
    tmp_file.parent.mkdir()

    checkpoint_dir = tmp_file.parent
    latest = None

    def checkpoint_callback(algorithm):
        nonlocal latest
        latest = algorithm.save(str(checkpoint_dir))

    trainer = Trainer(training_params, checkpoint_callback=checkpoint_callback)
    for _ in trainer.run():
        pass

    new_trainer = Trainer(training_params, checkpoint_path=latest)

    assert str(trainer.algorithm.get_policy().get_weights()) == str(
        new_trainer.algorithm.get_policy().get_weights()
    ), "weights not loaded properly"


@pytest.mark.parametrize(
    "fake_env",
    [
        (SpaceParam(Box(-1, 1, (84, 84, 4)), Discrete(5), [1]), 5),
    ],
    indirect=True,
)
@pytest.mark.parametrize("construct_training_config", [dqn.DQN], indirect=True)
def test_pretraining_with_bc(construct_training_config, bc_training_params):
    trainer = Trainer(bc_training_params)
    for _ in trainer.run():
        pass
    bc_weights = trainer.algorithm.get_policy().get_weights()

    config, _ = construct_training_config
    new_trainer = Trainer(config, pre_trained_weights=bc_weights)

    assert str(bc_weights) == str(
        new_trainer.algorithm.get_policy().get_weights()
    ), "weights do not match"
