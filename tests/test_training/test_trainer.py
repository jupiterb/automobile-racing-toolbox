import pytest
from racing_toolbox.training import Trainer
from racing_toolbox.training.config import TrainingParams
from tests.test_training.conftest import SpaceParam
from gym.spaces import Box, Discrete
from copy import deepcopy


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
        **training_config.dict()
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
        **training_config.dict()
    )
    training_params.num_rollout_workers = 0
    training_params.max_iterations = 2
    training_params.train_batch_size = 1
    training_params.checkpoint_frequency = 1

    tmp_file = tmp_path / "checkpoint_dir/foo"
    tmp_file.parent.mkdir()

    checkpoint_dir = tmp_file.parent
    checkpoint_callback = lambda algorithm: algorithm.save(str(checkpoint_dir))
    trainer = Trainer(training_params, checkpoint_callback=checkpoint_callback)
    trainer.run()

    latest_checkpoint = sorted(
        [d for d in checkpoint_dir.glob("checkpoint*") if d.is_dir()],
        key=lambda path: str(path),
    )[-1]
    print(latest_checkpoint)
    new_trainer = Trainer(training_params, checkpoint_path=latest_checkpoint)

    assert str(trainer.algorithm.get_state()) == str(
        new_trainer.algorithm.get_state()
    ), "weights not loaded properly"
