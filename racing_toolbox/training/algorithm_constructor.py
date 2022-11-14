from ray.rllib import algorithms as alg
from ray.rllib.algorithms import dqn

from ray.rllib.algorithms import dqn
from ray.rllib.algorithms import bc
from racing_toolbox.training.config import DQNConfig, BCConfig
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.training.config.user_defined import AlgorithmConfig
from ray.rllib.offline.estimators import ImportanceSampling


__config_to_cls_map: dict[type[AlgorithmConfig], type[alg.AlgorithmConfig]] = {
    DQNConfig: dqn.DQNConfig,
    BCConfig: bc.BCConfig,
}


def construct_cls(config: TrainingParams) -> alg.Algorithm:
    conf_cls = __config_to_cls_map[type(config.algorithm)]
    algo_conf = (
        conf_cls()
        .environment(
            env=config.env_name,
            observation_space=config.observation_space,
            action_space=config.action_space,
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
        .offline_data(
            input_=config.offline_data if config.offline_data else config.input_
        )
    )
    if config.evaluation_config is not None:
        conf = config.evaluation_config
        algo_conf.evaluation(
            evaluation_interval=conf.eval_interval_frequency,
            evaluation_duration=conf.eval_duration,
            evaluation_duration_unit=conf.eval_duration_unit,
            off_policy_estimation_methods={
                conf.eval_name: {"type": ImportanceSampling}
            },
        )
    if hasattr(algo_conf, "replay_buffer_config"):
        buffer_conf = algo_conf.replay_buffer_config.update(
            **config.algorithm.replay_buffer_config.dict()
        )
        algo_conf.training(replay_buffer_config=buffer_conf)
    return algo_conf.build()
