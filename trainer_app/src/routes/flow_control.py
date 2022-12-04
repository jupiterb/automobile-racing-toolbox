from fastapi import APIRouter, Depends
from trainer_app.src.schemas import StartTaskRequest
from trainer_app.src.tasks import app, start_training_task
from trainer_app.src.const import EnvVarsConfig
from celery.task.control import revoke 


flow_router = APIRouter()


@flow_router.put("/start")
def start_training(body: StartTaskRequest, env_vars: EnvVarsConfig = Depends(EnvVarsConfig)):
    """run training task"""
    result = start_training_task.delay(
        body.training_config,
        body.game_config,
        body.env_config,
        env_vars.default_policy_host,
        env_vars.default_policy_port,
        body.checkpoint_reference,
        None        
    )
    return result.id


@flow_router.get("/stop/{task_id}")
def stop_training(task_id):
    """stop running training task"""
    revoke(task_id, terminate=True)
    