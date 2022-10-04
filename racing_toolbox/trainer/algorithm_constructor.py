from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms import dqn

from racing_toolbox.trainer.config import DQNConfig
from racing_toolbox.trainer.config.params import TrainingParams

__config_to_cls_map = {DQNConfig: dqn.DQNConfig}


def construct_cls(config: TrainingParams) -> Algorithm:
    conf_cls = __config_to_cls_map.get(type(config.algorithm))
    algo_conf = (
        conf_cls()
        .environment(
            observation_space=config.env.observation_space,
            action_space=config.env.action_space,
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
        .offline_data(input_=config.input_)
    )
    if hasattr(algo_conf, "replay_buffer_config"):
        buffer_conf = algo_conf.replay_buffer_config.update(
            **config.algorithm.replay_buffer_config.dict()
        )
        algo_conf.training(replay_buffer_config=buffer_conf)
    return algo_conf.build()
