import argparse
import json
import gym
import numpy as np

from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.environment.builder import wrapp_env

from racing_toolbox.training import Trainer
from racing_toolbox.training.config import (
    TrainingConfig,
    BCConfig,
    EvalConfig,
    ModelConfig,
    TrainingParams,
)


def main():
    args = get_cli_args()

    env_config: EnvConfig
    with open(args.env_config) as ep:
        env_config = EnvConfig(**json.load(ep))

    empty = Empty()
    empty.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(*env_config.observation_config.shape, 3),
        dtype=np.uint8,
    )
    wrapped = wrapp_env(empty, env_config)

    training_config = get_train_config(
        wrapped.observation_space, wrapped.action_space, args.dataset
    )
    trainer = Trainer(training_config)
    trainer.run()


class Empty(gym.Env):
    def __init__(self) -> None:
        super().__init__()


def get_train_config(obs_space, act_space, dataset_path) -> TrainingParams:
    eval_conf = EvalConfig(
        eval_name="test_evaluation",
        eval_interval_frequency=1,
        eval_duration_unit="timesteps",
        eval_duration=10,
    )

    training_config: TrainingConfig = TrainingConfig(
        num_rollout_workers=0,
        rollout_fragment_length=10,
        train_batch_size=12,
        max_iterations=20,
        algorithm=BCConfig(),
        evaluation_config=eval_conf,
        model=ModelConfig(
            fcnet_hiddens=[50],
            fcnet_activation="relu",
            conv_filters=[
                (32, (8, 8), 4),
                (64, (4, 4), 2),
                (64, (3, 3), 1),
                (64, (8, 8), 1),
            ],
        ),
    )

    params = TrainingParams(
        **training_config.dict(),
        input_=[dataset_path],
        observation_space=obs_space,
        action_space=act_space,
    )
    return params


def get_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env_config",
        type=str,
        help="Path to json with env configuartion",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to json with OCR configuartion",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
