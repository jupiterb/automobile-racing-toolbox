from pydantic import BaseModel
from typing import Optional, Callable
from ray.rllib.algorithms import Algorithm
import wandb, os
from pathlib import Path
from ray.rllib.env.policy_server_input import PolicyServerInput
import boto3
from racing_toolbox.datatool.services.s3 import S3Dataset
from racing_toolbox.datatool.container import DatasetContainer
from src.const import EnvVarsConfig, TMP_DIR
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.environment.mocked import MockedEnv
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig
from racing_toolbox.environment import builder
import logging
from celery import Celery
import torch.utils.data as th_data
import numpy as np
import torch as th

logger = logging.getLogger(__name__)


class WorkerFailure(Exception):
    def __init__(self, worker_address: str, reason: str, details: Optional[str] = None):
        super().__init__()
        self.worker_address = worker_address
        self.reason = reason
        self.details = details


def make_celery(config: EnvVarsConfig, port: int, name: str):
    class CeleryConfig:
        task_serializer = "pickle"
        result_serializer = "pickle"
        event_serializer = "json"
        accept_content = ["application/json", "application/x-python-serialize"]
        result_accept_content = ["application/json", "application/x-python-serialize"]

    print(config.celery_broker_url + f"/{port}", name)

    celery = Celery(
        name,
        broker=config.celery_broker_url + f"/{port}",
        backend=config.celery_backend_url,
    )
    celery.conf.result_extended = True
    celery.config_from_object(CeleryConfig)
    return celery


def wandb_checkpoint_callback_factory(checkpoint_name: str, dir: Path):
    def callback(algorithm: Algorithm):
        nonlocal checkpoint_name

        name = checkpoint_name.split(":")[0]
        print(f"checkpoint name: {name}")
        checkpoint_artifact = wandb.Artifact(name, type="checkpoint")
        chkpnt_path = algorithm.save(str(dir))
        checkpoint_artifact.add_dir(chkpnt_path, name="checkpoint")
        wandb.log_artifact(checkpoint_artifact)

    return callback


def log_config(config: BaseModel, name: str) -> None:
    filename = Path(name + ".json")
    with open(filename, "w") as f:
        f.write(config.json())
    wandb.save(str(filename), policy="now")
    # filename.unlink()


def get_training_params(
    host: str,
    port: int,
    training_config: TrainingConfig,
    game_config: GameConfiguration,
    env_config: EnvConfig,
):
    def input_(ioctx):
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                "0.0.0.0",
                port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None

    # TODO: How to choose correct interface action mapping based only on game config?
    mocked_env = MockedEnv(
        game_config.discrete_actions_mapping, game_config.window_size
    )
    env = builder.wrapp_env(mocked_env, env_config)
    print(f"observation space: {env.observation_space.shape}")
    logger.warning(
        f"observation space shape: {env.observation_space.shape}, {env.observation_space.low} - {env.observation_space.high}"
    )
    trainer_params = TrainingParams(
        **training_config.dict(),
        observation_space=env.observation_space,
        action_space=env.action_space,
        input_=input_,
    )
    return trainer_params


def tensordataset_from_bucket(
    recordings_refs: list[str],
    bucket_name: str,
    aws_key: str,
    aws_secret_key: str,
    transforms: Callable[[np.ndarray], th.Tensor],
) -> th_data.TensorDataset:
    container = DatasetContainer()
    for file_ref in recordings_refs:
        dataset = S3Dataset(bucket_name, file_ref, aws_key, aws_secret_key)
        assert container.try_add(
            dataset
        ), f"Dataset {file_ref} is incompatible with the rest"

    observations: list[th.Tensor] = []
    for observation, _ in container.get_all():
        observations.append(transforms(observation))

    return th_data.TensorDataset(th.stack(observations))
