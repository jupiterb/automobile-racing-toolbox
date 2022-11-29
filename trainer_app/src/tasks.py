from trainer_app.src.const import EnvVarsConfig
from racing_toolbox.training import Trainer
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.environment.config.env import EnvConfig
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

logger = get_task_logger(__name__)


def make_celery(config: EnvVarsConfig):
    celery = Celery(
        "tasks", broker=config.celery_broker_url, backend=config.celery_backend_url
    )
    return celery


app = make_celery(EnvVarsConfig())


@app.task(bind=True, base=AbortableTask)
def start_training(
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

    env = builder.setup_env(game_config, env_config)
    trainer_params = TrainingParams(
        **training_config.dict(),
        observation_space=env.observation_space,
        action_space=env.action_space,
        input_=input_,
    )
    trainer = Trainer(trainer_params, checkpoint_path, pre_trained_weights)
    with wandb.init(project="ART", name=f"task_{self.}") as run:
        game_conf_artifact = wandb.Artifact("game-config", type="config")
        env_conf_artifact = wandb.Artifact("env-config", type="config")
        training_conf_artifact = wandb.Artifact("training-config", type="config")
        game_conf_artifact.add(game_config.dict(), "latest")
        env_conf_artifact.add(env_config.dict(), "latest")
        training_conf_artifact.add(TrainingConfig(**config.dict()).dict(), "latest")
        run.log_artifact(training_conf_artifact)
        run.log_artifact(game_conf_artifact)
        run.log_artifact(training_conf_artifact)

        for epoch in trainer.run():
            logger.debug(f"epoch {epoch} done")
            if self.is_aborted():
                logger.warning("task has been aborted")
                break

