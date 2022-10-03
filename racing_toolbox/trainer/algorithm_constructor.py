import inspect
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms import dqn

from racing_toolbox.trainer.config import DQNConfig, TrainingConfig

__config_to_cls_map = {DQNConfig: dqn.DQNConfig}


def construct_cls(config: TrainingConfig) -> Algorithm:
    conf_cls = __config_to_cls_map.get(type(config.algorithm))
    algo_conf = (
        conf_cls()
        .environment(
            env=config.env,
        )
        .framework(framework="torch")
        .rollouts(
            num_rollout_workers=config.num_rollout_workers,
            rollout_fragment_length=config.rollout_fragment_length,
            compress_observations=config.compress_observations,
        )
        .training(
            gamma=config.gamma,
            lr=config.lr,
            train_batch_size=config.train_batch_size,
            **config.algorithm.dict(),
            model=config.model.dict()
        )
    )
    if hasattr(algo_conf, "replay_buffer_config"):
        buffer_conf = algo_conf.replay_buffer_config.update(
            **config.algorithm.replay_buffer_config.dict()
        )
        algo_conf.training(replay_buffer_config=buffer_conf)
    return conf_cls.build()
