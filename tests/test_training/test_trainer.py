import pytest
from ray.rllib.algorithms import dqn, bc
from gym.spaces import Box, Discrete
from copy import deepcopy
from racing_toolbox.training import Trainer
from racing_toolbox.training.config import TrainingParams
import racing_toolbox.training.algorithm_constructor as algo
from tests.test_training.conftest import SpaceParam


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
def test_if_trainer_runs_properly(fake_env, training_config):
    training_params = TrainingParams(
        env_name=fake_env.name,
        observation_space=fake_env.observation_space,
        action_space=fake_env.action_space,
        **training_config.dict(),
    )
    training_params.num_rollout_workers = 0
    training_params.max_iterations = 2
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
    ],
    indirect=True,
)
def test_if_checkpoint_laoded(fake_env, training_config, tmp_path):
    training_params = TrainingParams(
        env_name=fake_env.name,
        observation_space=fake_env.observation_space,
        action_space=fake_env.action_space,
        **training_config.dict(),
    )
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
    trainer.run()

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
    trainer.run()
    bc_weights = trainer.algorithm.get_policy().get_weights()

    config, _ = construct_training_config
    new_trainer = Trainer(config, pre_trained_weights=bc_weights)

    assert str(bc_weights) == str(
        new_trainer.algorithm.get_policy().get_weights()
    ), "weights do not match"
