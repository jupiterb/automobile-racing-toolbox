from pydantic import BaseModel
from typing import Optional 
from ray.rllib.algorithms import Algorithm
import wandb
from pathlib import Path


class WorkerFailure(Exception):
    def __init__(self, worker_address: str, reason: str, details: Optional[str] = None):
        super().__init__()
        self.worker_address = worker_address
        self.reason = reason
        self.details = details


def wandb_checkpoint_callback_factory(checkpoint_artifact: wandb.Artifact, dir: Path):
    def callback(algorithm: Algorithm):
        chkpnt_path = algorithm.save(str(dir))
        checkpoint_artifact.add_dir(chkpnt_path, name="checkpoint")
        wandb.log_artifact(checkpoint_artifact)

    return callback


def log_config(config: BaseModel, name: str) -> None:
    unique_filename = Path(name + ".json")
    with open(unique_filename, "w") as f:
        f.write(config.json())
    wandb.save(str(unique_filename), policy="now")
    unique_filename.unlink()
