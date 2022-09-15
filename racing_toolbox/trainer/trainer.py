from multiprocessing import Process
import itertools as it
from ray.rllib.algorithms import Algorithm
from ray.tune.logger import pretty_print
import gym

from trainer import config


class Trainer(Process):
    def __init__(self, config):
        self.config = config

    @property
    def algorithm(self) -> Algorithm:
        if self._algorithm is None:
            self._algorithm = self._setup_algorithm(self.config.algorithm)
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

    def _setup_algorithm(self, conf: config.AlgorithmConfig) -> Algorithm:
        "initialize algorithm object from configuration"
        if isinstance(config, config.DQNConfig):
            from ray.rllib.algorithms import dqn

            dqn_conf = dqn.DQNConfig
            buffer_config = dqn_conf.replay_buffer_config.update(
                conf.replay_buffer_config
            )
            dqn_conf.training(
                **conf.dict(exclude={"replay_buffer_config"}),
                replay_buffer_config=buffer_config
            ).environment(self._env)
            return dqn.DQN(dqn_conf)
        else:
            raise NotImplementedError

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
