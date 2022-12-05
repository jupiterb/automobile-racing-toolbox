from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from trainer_app.src.schemas import StartTrainingRequest
from trainer_app.src.tasks.tasks import app, start_training_task, sync_workers, notify_workers, load_pretrained_weights
from trainer_app.src.const import EnvVarsConfig
from trainer_app.src.worker_registry import (
    RemoteWorkerRegistry,
    RemoteWorkerRef,
    MemoryRegistry,
)
from trainer_app.src.worker_registry.in_memory_registry import get_registry


flow_router = APIRouter()


@flow_router.put("/continue")
def continue_training():
    ...

@flow_router.put("/start")
def start_training(body: StartTrainingRequest, env_vars: EnvVarsConfig = Depends(EnvVarsConfig), registry: RemoteWorkerRegistry = Depends(get_registry)):
    """run training task"""
    workers = [w for w in registry.get_workers(body.game_config.game_id) if w.available]
    if len(workers) < body.training_config.num_rollout_workers:
        return PlainTextResponse(status_code=404, content="too many workers requested")
    workers = workers[:body.training_config.num_rollout_workers]    
    sync_task = sync_workers.s(
        workers=workers, 
        game_config=body.game_config,
        env_config=body.env_config, 
        host=env_vars.default_policy_host,
        port=env_vars.default_policy_port,
        wandb_project="ART",
        wandb_api_key=body.wandb_api_key,
        wandb_group=body.wandb_group
    )
    start_task = start_training_task.s(
        wandb_api_key=body.wandb_api_key,
        training_config=body.training_config,
        game_config=body.game_config,
        env_config=body.env_config,
        group=body.wandb_group,
        host=env_vars.default_policy_host,
        port=env_vars.default_policy_port,
        workers_ref=workers
    ).on_error(notify_workers.s([w.address for w in workers], "/stop"))
    if body.run_reference and body.checkpoint_name:
        load_weights_task = load_pretrained_weights.s(body.wandb_api_key, body.run_reference, body.checkpoint_name)
        result = (sync_task | load_weights_task | start_task)()
    else:
        start_task = start_training_task.s(pretrained_weights=None , **start_task.kwargs)
        result = (sync_task | start_task)()
    return result.id


@flow_router.get("/stop/{task_id}")
def stop_training(task_id):
    """stop running training task"""
    start_training_task.AsyncResult(task_id).abort()
    