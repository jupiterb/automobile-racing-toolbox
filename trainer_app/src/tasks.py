from trainer_app.src.const import EnvVarsConfig, TMP_DIR
from racing_toolbox.training import Trainer
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.environment.mocked import MockedEnv
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig
from racing_toolbox.environment import builder
from typing import Optional, Callable, Any
from pathlib import Path
from celery import Celery, Task
from celery.contrib.abortable import AbortableTask
from celery.utils.log import get_task_logger
from ray.rllib.env.policy_server_input import PolicyServerInput
import os
import wandb
import json
import uuid
from pydantic import BaseModel
from ray.rllib.algorithms import Algorithm


logger = get_task_logger(__name__)


def make_celery(config: EnvVarsConfig):
    celery = Celery(
        "tasks", broker=config.celery_broker_url, backend=config.celery_backend_url
    )
    return celery


app = make_celery(EnvVarsConfig())


@app.task(bind=True, base=AbortableTask)
def start_training_task(
    self: AbortableTask,
    wandb_api_key: str,
    training_config: TrainingParams,
    game_config: GameConfiguration,
    env_config: EnvConfig,
    host: str,
    port: int,
    checkpoint_path: Optional[Path] = None,
    pre_trained_weights: Optional[dict] = None,
):
    def input_(ioctx):
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                host,
                port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None

    os.environ["WANDB_API_KEY"] = wandb_api_key

    # TODO: How to choose correct interface action mapping based only on game config?
    mocked_env = MockedEnv(game_config.discrete_actions_mapping, game_config.window_size)
    env = builder.wrapp_env(mocked_env, env_config)
    trainer_params = TrainingParams(
        **training_config.dict(),
        observation_space=env.observation_space,
        action_space=env.action_space,
        input_=input_,
    )
    with wandb.init(project="ART", name=f"task_{self.request.id}") as run:
        _log_config(game_config)
        _log_config(env_config)
        _log_config(training_config)

        chkpnt_dir = TMP_DIR / f"checkpoints_{run.id}"
        chkpnt_dir.mkdir()
        chkpnt_artifact = wandb.Artifact(f"checkpoint-{run.id}", type="checkpoint")
        checkpoint_callback = _wandb_checkpoint_callback_factory(
            chkpnt_artifact, chkpnt_dir
        )
        trainer = Trainer(
            trainer_params,
            checkpoint_path,
            pre_trained_weights,
            checkpoint_callback=checkpoint_callback,
        )

        for epoch in trainer.run():
            logger.debug(f"epoch {epoch} done")
            if self.is_aborted():
                logger.warning("task has been aborted")
                break


def _wandb_checkpoint_callback_factory(checkpoint_artifact: wandb.Artifact, dir: Path):
    def callback(algorithm: Algorithm):
        chkpnt_path = algorithm.save(str(dir))
        checkpoint_artifact.add_dir(chkpnt_path, name="checkpoint")
        wandb.log_artifact(checkpoint_artifact)

    return callback


def _log_config(config: BaseModel) -> None:
    unique_filename = Path(str(uuid.uuid4()) + ".json")
    with open(unique_filename, "w") as f:
        f.write(config.json())
    wandb.save(str(unique_filename), policy="now")
    unique_filename.unlink()
