import itertools as it
import logging
from pathlib import Path
from typing import Optional
from ray.rllib.algorithms import Algorithm
from ray.air._internal.json import SafeFallbackEncoder
import json
import ray
import wandb

import racing_toolbox.trainer.algorithm_constructor as algo
from racing_toolbox.trainer.config.params import TrainingParams

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: TrainingParams,
        checkpoint_path: Optional[Path] = None,
        pre_trained_weights: Optional[dict] = None,
    ):
        self.config = config
        # lazy values
        self._algorithm: Algorithm = algo.construct_cls(config)
        if pre_trained_weights:
            self._algorithm.get_policy().set_weights(pre_trained_weights)
        if checkpoint_path:
            logger.info(f"restoring from checkpoint {checkpoint_path}")
            self._algorithm.restore(str(checkpoint_path))

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    def run(self) -> None:
        "training loop"
        ray.init(ignore_reinit_error=True)

        for i in it.count(0, 1):
            results = self.algorithm.train()
            self.make_checkpoint()
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

    def make_checkpoint(self) -> None:
        ckpnt_path = self.algorithm.save()
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
