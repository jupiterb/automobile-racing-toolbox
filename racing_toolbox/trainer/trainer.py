import itertools as it
from pathlib import Path
from typing import Optional
from ray.rllib.algorithms import Algorithm
from ray.tune.logger import pretty_print

from racing_toolbox.trainer.config import TrainingConfig
import racing_toolbox.trainer.algorithm_constructor as algo


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        # lazy values
        self._algorithm: Algorithm = algo.construct_cls(config)

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

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
