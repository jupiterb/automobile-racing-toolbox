from multiprocessing import Process

from racing_toolbox.trainer.config import AlgorithmConfig


class Trainer(Process):
    
    def run(self) -> None:
        "training loop"

    def _stop_criterion(self) -> bool:
        "decide wether to stop training loop"

    def _setup_algorithm(self, config: AlgorithmConfig) -> None:
        "initialize algorithm object from configuration"

    def _cleanup(self) -> bool:
        "make sure proper checkpoints were saved"

    def _sync_loop(self) -> None:
        "wait for connection from client to sync env configuration"

    def _log(self, metrics: dict) -> None:
        "log metrics, models, videos to wandb"

    def _checkpoint(self, model) -> None:
        "save current model"
