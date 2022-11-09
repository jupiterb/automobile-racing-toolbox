import gym
import logging

from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.offline.estimators import ImportanceSampling

from racing_toolbox.trainer.config import UserDefinedBCConfig

logger = logging.getLogger(__name__)


def train_bc(config: UserDefinedBCConfig, env: gym.Env, path_to_data: str):
    algo_config = (
        BCConfig()
        .evaluation(
            evaluation_interval=2,
            evaluation_duration=5,
            off_policy_estimation_methods={"bc_eval": {"type": ImportanceSampling}},
        )
        .environment(
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        .framework(framework="torch")
        .training(
            lr=config.lr,
            train_batch_size=int(config.train_batch_size),
            model=config.model.dict(),
        )
        .offline_data(input_=[path_to_data])
    )

    behavior = algo_config.build()

    for i in range(config.num_iterations):
        print(f"BC iteration {i}")
        results = behavior.train()

        try:
            bc_val_results = results["evaluation"]["off_policy_estimator"]["bc_eval"]
            print(bc_val_results)
        except:
            pass

    return behavior
