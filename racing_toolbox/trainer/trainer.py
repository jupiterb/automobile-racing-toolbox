from multiprocessing import Process
import itertools as it
from typing import Optional
from ray.rllib.algorithms import Algorithm
from ray.tune.logger import pretty_print
import gym

from trainer import config


class Trainer:
    def __init__(self, config):
        self.config = config
        # lazy values
        self._algorithm: Optional[Algorithm] = None

    @property
    def algorithm(self) -> Algorithm:
        if self._algorithm is None:
            self._algorithm = self._setup_algorithm()
        return self.__algorithm

    def run(self) -> None:
        "training loop"

        for i in it.count(0, 1):
            results = self.algorithm.train()
            checkpoint_dir_path = self.algorithm.save()
            self._log(results)
            if self._stop_criterion(results, i):
                break
        self._cleanup()

    def _stop_criterion(self, metrics, iter) -> bool:
        "decide wether to stop training loop"
        return (
            metrics["episode_reward_mean"] >= self._stop_reward
            or iter >= self._max_iterations
        )

    def _setup_algorithm(self) -> Algorithm:
        "initialize algorithm object from configuration"
        if isinstance(self.config.algorithm, config.DQNConfig):
            from ray.rllib.algorithms import dqn

            algo_conf = self.config.algorithm

            dqn_conf = dqn.DQNConfig()
            buffer_config = dqn_conf.replay_buffer_config.update(
                algo_conf.replay_buffer_config
            )
            return (
                dqn_conf.training(
                    **algo_conf.dict(exclude={"replay_buffer_config"}),
                    replay_buffer_config=buffer_config
                )
                .environment(
                    observation_space=self.config.env.observation_space,
                    action_space=self.config.env.action_space,
                )
                .build()
            )
        else:
            raise NotImplementedError(type(self.config.algorithm))

    def _cleanup(self) -> bool:
        "make sure proper checkpoints were saved"
        print("cleanup")

    def _sync_loop(self) -> None:
        "wait for connection from client to sync env configuration"

    def _log(self, metrics: dict) -> None:
        "log metrics, models, videos to wandb"
        print(pretty_print(metrics))

    def _checkpoint(self, model) -> None:
        "save current model"
