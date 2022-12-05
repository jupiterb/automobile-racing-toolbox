from trainer_app.src.const import EnvVarsConfig, TMP_DIR
from trainer_app.src.worker_registry import RemoteWorkerRef
from trainer_app.src.tasks.utils import  WorkerFailure, wandb_checkpoint_callback_factory, log_config
from racing_toolbox.training import Trainer
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.environment.mocked import MockedEnv
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig
from racing_toolbox.environment import builder
from typing import Optional
from pathlib import Path
from celery import Celery
from celery.contrib.abortable import AbortableTask
from celery.utils.log import get_task_logger
from ray.rllib.env.policy_server_input import PolicyServerInput
import os
import wandb
from ray.rllib.algorithms import Algorithm
import grequests


logger = get_task_logger(__name__)


def make_celery(config: EnvVarsConfig):
    celery = Celery(
        "tasks", broker=config.celery_broker_url, backend=config.celery_backend_url
    )
    celery.conf.result_extended = True
    return celery


app = make_celery(EnvVarsConfig())


@app.task(bind=True, result_extended=True, base=AbortableTask)
def start_training_task(
    self: AbortableTask,
    pretrained_weights: Optional[dict],
    wandb_api_key: str,
    training_config: TrainingParams,
    game_config: GameConfiguration,
    env_config: EnvConfig,
    group: str,
    host: str,
    port: int,
    workers_ref: list[RemoteWorkerRef]
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
    mocked_env = MockedEnv(
        game_config.discrete_actions_mapping, game_config.window_size
    )
    env = builder.wrapp_env(mocked_env, env_config)
    trainer_params = TrainingParams(
        **training_config.dict(),
        observation_space=env.observation_space,
        action_space=env.action_space,
        input_=input_,
    )
    with wandb.init(project="ART", name=f"task_{self.request.id}", group=group) as run:
        log_config(game_config, "game_config")
        log_config(env_config, "env_config")
        log_config(training_config, "training_config")

        chkpnt_dir = TMP_DIR / f"checkpoints_{run.id}"
        chkpnt_dir.mkdir()
        chkpnt_artifact = wandb.Artifact(f"checkpoint-{run.id}", type="checkpoint")
        checkpoint_callback = wandb_checkpoint_callback_factory(
            chkpnt_artifact, chkpnt_dir
        )
        trainer = Trainer(
            trainer_params,
            pre_trained_weights=pretrained_weights,
            checkpoint_callback=checkpoint_callback,
        )
        notify_workers([w.address for w in workers_ref], "/start")
        for epoch in trainer.run():
            logger.debug(f"epoch {epoch} done")
            if self.is_aborted():
                logger.warning("task has been aborted")
                break
        notify_workers([w.address for w in workers_ref], "/stop")

@app.task 
def load_pretrained_weights(wandb_api_key: str, run_ref: str, checkpoint_name: str):
    import wandb
    os.environ["WANDB_API_KEY"] = wandb_api_key
    run = wandb.init(project="ART")
    model_art = run.use_artifact(f"{run_ref}/{checkpoint_name}", type='model')
    checkpoint_dir = model_art.download()
    game_config = GameConfiguration.parse_raw(wandb.restore("game_config.json"))
    env_config = EnvConfig.parse_raw(wandb.restore("env_config.json"))
    training_config = TrainingConfig.parse_raw(wandb.restore("training_config.json"))

    mocked_env = MockedEnv(
        game_config.discrete_actions_mapping, game_config.window_size
    )
    env = builder.wrapp_env(mocked_env, env_config)
    trainer_params = TrainingParams(
        **training_config.dict(),
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    algorithm: Algorithm = Trainer(trainer_params, checkpoint_path=Path(checkpoint_dir)).algorithm
    return algorithm.get_policy().get_weights()

    
@app.task(ignore_result=True)
def notify_workers(urls: list[str], route: str):
    rs = (grequests.get(u + route) for u in urls)
    responses = grequests.map(rs)
    for r, addr in zip(responses, urls):
        if r.status_code != 200:
            raise WorkerFailure(addr, "Cannot start")


@app.task(ignore_result=True)
def sync_workers(
    workers: list[RemoteWorkerRef],
    game_config: GameConfiguration,
    env_config: EnvConfig,
    host: str,
    port: int,
    wandb_project: str,
    wandb_api_key: str,
    wandb_group: str,
):
    urls = [w.address for w in workers]
    rs = []
    for i, url in enumerate(urls):
        body = {
            "game_config": game_config,
            "env_config": env_config,
            "policy_address": f"http://{host}:{port + i}",
            "wandb_project": wandb_project,
            "wandb_api_key": wandb_api_key,
            "wandb_group": wandb_group,
        }
        rs.append(grequests.post(url, data=body))
    responses = grequests.map(rs)
    for r, addr in zip(responses, urls):
        if r.status_code != 200:
            raise WorkerFailure(addr, "Cannot sync")
