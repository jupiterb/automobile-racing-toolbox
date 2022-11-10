import itertools as it
import logging
from pathlib import Path
from typing import Optional, Callable, Any
from ray.rllib.algorithms import Algorithm
from ray.air._internal.json import SafeFallbackEncoder
import json
import ray
import wandb

import racing_toolbox.training.algorithm_constructor as algo
from racing_toolbox.training.config.params import TrainingParams

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: TrainingParams,
        checkpoint_path: Optional[Path] = None,
        pre_trained_weights: Optional[dict] = None,
        checkpoint_callback: Optional[Callable[[Algorithm], Any]] = None,
    ):
        self.config = config
        # lazy values
        self._algorithm: Algorithm = algo.construct_cls(config)
        if pre_trained_weights:
            self._algorithm.get_policy().set_weights(pre_trained_weights)
        if checkpoint_path:
            logger.info(f"restoring from checkpoint {checkpoint_path}")
            self._algorithm.restore(str(checkpoint_path))

        self._checkpoint_callback = checkpoint_callback or self.make_checkpoint

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    def run(self) -> None:
        "training loop"
        ray.init(ignore_reinit_error=True)

        for i in it.count(0, 1):
            results = self.algorithm.train()
            if i + 1 % self.config.checkpoint_frequency == 0:
                self._checkpoint_callback(self.algorithm)
            self._log(results)
            if self._stop_criterion(results, i):
                logger.info("Stop criterion satisfied. Exiting training loop.")
                break

    def _stop_criterion(self, metrics, iter) -> bool:
        "decide wether to stop training loop"
        return (
            metrics["episode_reward_mean"] >= self.config.stop_reward
            or iter >= self.config.max_iterations
        )

    def _log(self, raw_logs: dict) -> None:
        "log metrics, models, videos to wandb"
        stats = self._get_stats(raw_logs)
        if wandb.run is not None:
            wandb.log(stats)
        logger.debug(stats)
        logger.info(self._filter_and_format_stats(stats))

    @staticmethod
    def make_checkpoint(algorithm: Algorithm) -> None:
        ckpnt_path = algorithm.save()
        logger.info(f"Saved checkpoint to: {ckpnt_path}")

    @staticmethod
    def _get_stats(result):
        result = result.copy()
        result.update(config=None)  # drop config from pretty print
        result.update(hist_stats=None)  # drop hist_stats from pretty print
        out = {}
        for k, v in result.items():
            if v is not None:
                out[k] = v

        cleaned = json.dumps(out, cls=SafeFallbackEncoder)
        return json.loads(cleaned)

    @staticmethod
    def _filter_and_format_stats(stats: dict) -> str:
        rw_mean = stats.get("episode_reward_mean")
        return f"episode_reward_mean: {rw_mean}"
