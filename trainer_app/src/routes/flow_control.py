from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from src.schemas import StartTrainingRequest, ResumeTrainingRequest
from src.tasks import tasks 
from src.const import EnvVarsConfig
from src.worker_registry import (
    RemoteWorkerRegistry,
)
from src.worker_registry.in_memory_registry import get_registry
from logging import getLogger
logger = getLogger(__name__)

flow_router = APIRouter()


@flow_router.put("/resume")
def continue_training(
    body: ResumeTrainingRequest,
    registry: RemoteWorkerRegistry = Depends(get_registry),
):
    env_vars: EnvVarsConfig = EnvVarsConfig()
    workers = [w for w in registry.get_workers(body.game_id) if w.available]
    if len(workers) < body.training_config.num_rollout_workers:
        return PlainTextResponse(status_code=404, content="too many workers requested")
    workers = workers[: body.training_config.num_rollout_workers]
    resume_task = tasks.continue_training_task.delay(
        training_config=body.training_config,
        wandb_api_key=body.wandb_api_key,
        run_ref=body.wandb_run_reference,
        checkpoint_name=body.checkpoint_name,
        host=env_vars.default_policy_host,
        port=env_vars.default_policy_port,
        workers_ref=workers
    )
    return resume_task.id


@flow_router.put("/start")
def start_training(
    body: StartTrainingRequest,
    registry: RemoteWorkerRegistry = Depends(get_registry),
):
    logger.info("tu")
    """run training task"""
    env_vars: EnvVarsConfig = EnvVarsConfig()
    workers = [w for w in registry.get_workers(body.game_config.game_id) if w.available]
    if len(workers) < body.training_config.num_rollout_workers:
        return PlainTextResponse(status_code=404, content="too many workers requested")
    
    workers = workers[: body.training_config.num_rollout_workers]
    sync_task = tasks.sync_workers.s(
        workers=workers,
        game_config=body.game_config,
        env_config=body.env_config,
        host=env_vars.default_policy_host,
        port=env_vars.default_policy_port,
        wandb_project="ART",
        wandb_api_key=body.wandb_api_key,
        wandb_group=body.wandb_group,
    )
    start_task = tasks.start_training_task.s(
        wandb_api_key=body.wandb_api_key,
        training_config=body.training_config,
        game_config=body.game_config,
        env_config=body.env_config,
        group=body.wandb_group,
        host=env_vars.default_policy_host,
        port=env_vars.default_policy_port,
        workers_ref=workers,
    )#.on_error(tasks.notify_workers.s(urls=[w.address for w in workers], route="/worker/stop"))

    if body.run_reference and body.checkpoint_name:
        load_weights_task = tasks.load_pretrained_weights.s(
            body.wandb_api_key, body.run_reference, body.checkpoint_name
        )
        result = (sync_task | load_weights_task | start_task)()
    else:
        result = (sync_task | start_task)()

    return result.id


@flow_router.get("/stop/{task_id}")
def stop_training(task_id):
    """stop running training task"""
    tasks.app.Task().AsyncResult(task_id).abort()

@flow_router.post("/probe")
def start_probe():
    result = tasks.probe_task.delay()
    return result.id
