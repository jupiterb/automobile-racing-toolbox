from src.const import EnvVarsConfig, TMP_DIR
from src.worker_registry import RemoteWorkerRef
from src.tasks.utils import (
    WorkerFailure,
    wandb_checkpoint_callback_factory,
    log_config,
    get_training_params,
)
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
from celery.utils.log import get_task_logger, base_logger
from celery.signals import after_setup_task_logger, setup_logging
from celery.app.log import TaskFormatter
import os, uuid
import wandb
from ray.rllib.algorithms import Algorithm
import requests
import logging 
logging.basicConfig(format='%(levelname)-8s: %(asctime)s - %(name)s.%(funcName)s() - %(message)s')

logger = get_task_logger(__name__)

def make_celery(config: EnvVarsConfig):

    class CeleryConfig:
        task_serializer = "pickle"
        result_serializer = "pickle"
        event_serializer = "json"
        accept_content = ["application/json", "application/x-python-serialize"]
        result_accept_content = ["application/json", "application/x-python-serialize"]


    celery = Celery(
        "tasks", broker=config.celery_broker_url, backend=config.celery_backend_url
    )
    celery.conf.result_extended = True
    celery.config_from_object(CeleryConfig)
    return celery


app = make_celery(EnvVarsConfig())

@setup_logging.connect
def setup_celery_logging(**kwargs):
    pass


class TrainingTask(AbortableTask):
    def training_loop(
        self,
        game_config: GameConfiguration,
        env_config: EnvConfig,
        training_config: TrainingConfig,
        trainer_params: TrainingParams,
        pretrained_weights: Optional[dict],
        checkpoint_dir: Optional[Path],
        workers_ref: list[RemoteWorkerRef],
        wandb_run=None
    ):
        run = wandb_run or wandb.run
        print("going to log configs")
        log_config(game_config, "game_config")
        log_config(env_config, "env_config")
        log_config(training_config, "training_config")
        print("going to build trainer")
        trainer = Trainer(
            trainer_params,
            pre_trained_weights=pretrained_weights,
            checkpoint_callback=self._get_calllback(run),
            checkpoint_path=checkpoint_dir,
        )
        print("going to notify workers about start")
        notify_workers(urls=[w.address for w in workers_ref], route="/worker/start", method="post")
        for epoch in trainer.run():
            logger.info(f"epoch {epoch} done")
            if self.is_aborted():
                logger.warning("task has been aborted")
                break
        notify_workers(urls=[w.address for w in workers_ref], route="/worker/stop", method="post")

    def _get_calllback(self, run):
        chkpnt_dir = TMP_DIR / f"checkpoints_{run.id}"
        chkpnt_dir.mkdir()
        chkpnt_artifact = wandb.Artifact(f"checkpoint-{run.id}", type="checkpoint")
        checkpoint_callback = wandb_checkpoint_callback_factory(
            chkpnt_artifact, chkpnt_dir
        )
        return checkpoint_callback


@app.task(bind=True, result_extended=True, base=TrainingTask)
def start_training_task(
    self: TrainingTask,
    pretrained_weights: Optional[dict],
    wandb_api_key: str,
    training_config: TrainingConfig,
    game_config: GameConfiguration,
    env_config: EnvConfig,
    group: str,
    host: str,
    port: int,
    workers_ref: list[RemoteWorkerRef],
):
    print("starting")
    os.environ["WANDB_API_KEY"] = wandb_api_key
    trainer_params = get_training_params(
        host, port, training_config, game_config, env_config
    )
    print("starting wandb run")
    with wandb.init(project="ART", name=f"task_{self.request.id}", group=group) as run:
        self.training_loop(
            game_config,
            env_config,
            training_config,
            trainer_params,
            pretrained_weights,
            None,
            workers_ref,
        )


@app.task(bind=True, result_extended=True, base=TrainingTask)
def continue_training_task(
    self: TrainingTask,
    training_config: TrainingConfig,
    wandb_api_key: str,
    run_ref: str,
    checkpoint_name: str,
    group: str,
    host: str,
    port: int,
    workers_ref: list[RemoteWorkerRef],
):
    os.environ["WANDB_API_KEY"] = wandb_api_key
    with wandb.init(project="ART") as run:
        checkpoint_ref = f"{'/'.join(run_ref.split('/')[:-1])}/{checkpoint_name}"
        print(checkpoint_ref)
        checkpoint_artefact = run.use_artifact(
            checkpoint_ref, type="checkpoint"
        )
        checkpoint_dir = checkpoint_artefact.download()
        game_config = GameConfiguration.parse_file(wandb.restore("game_config.json", run_path=run_ref).name)
        env_config = EnvConfig.parse_file(wandb.restore("env_config.json", run_path=run_ref).name)

    trainer_params = get_training_params(
        host, port, training_config, game_config, env_config
    )
    sync_workers(
        workers_ref,
        game_config,
        env_config,
        host,
        port,
        "ART",
        wandb_api_key,
        str(uuid.uuid1()),
    )
    print(checkpoint_dir)
    with wandb.init(project="ART", name=f"task_{self.request.id}", group=group) as run:
        self.training_loop(
            game_config,
            env_config,
            training_config,
            trainer_params,
            None,
            Path(checkpoint_dir).absolute() / "checkpoint",
            workers_ref,
        )


@app.task
def load_pretrained_weights(*args, wandb_api_key: str, run_ref: str, checkpoint_name: str, **kwargs):
    import wandb

    os.environ["WANDB_API_KEY"] = wandb_api_key
    with wandb.init(project="ART") as run:
        checkpoint_artefact = run.use_artifact(
            f"{run_ref}/{checkpoint_name}", type="checkpoint"
        )
        checkpoint_dir = checkpoint_artefact.download()
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
    algorithm: Algorithm = Trainer(
        trainer_params, checkpoint_path=Path(checkpoint_dir).absolute() / "checkpoint"
    ).algorithm
    return algorithm.get_policy().get_weights()


@app.task(ignore_result=True)
def notify_workers(*args, urls: list[str], route: str, method: str="get", **kwargs):
    responses = [getattr(requests, method)(u + route) for u in urls]
    # responses = grequests.map(rs)
    for r, addr in zip(responses, urls):
        if r.status_code != 200:
            raise WorkerFailure(addr, str(r.content))

import orjson 

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
    responses = []
    for i, url in enumerate(urls):
        body = {
            "game_config": orjson.loads(game_config.json()),
            "env_config": orjson.loads(env_config.json()),
            "policy_address": [host, port + i],
            "wandb_project": wandb_project,
            "wandb_api_key": wandb_api_key,
            "wandb_group": wandb_group,
        }
        
        print(url)
        r = requests.post(url + "/worker/sync", json=body)
        responses.append(r)
        if r is None or r.status_code != 200:
            logger.error(f"cannot sync with {url}. got response {r.content if r is not None else r}")
            raise WorkerFailure(url, "Cannot sync")
    # responses = grequests.map(rs)
    logger.info(f"responses {responses}")
    # for r, addr in zip(responses, urls):

import time 
@app.task(ignore_results=True)
def probe_task():
    time.sleep(10)
    print(10)
    return 10
