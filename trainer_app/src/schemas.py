from pydantic import BaseModel, validator
from typing import Optional, Any
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig
import uuid 

class StartTrainingRequest(BaseModel):
    game_config: GameConfiguration
    env_config: EnvConfig
    training_config: TrainingConfig
    run_reference: Optional[str] = None 
    checkpoint_name: Optional[str] = None 
    wandb_api_key: str 
    wandb_group: str = str(uuid.uuid1()) 


class ContinueTrainingRequest(BaseModel):
    wandb_run_reference: str 
    wandb_api_key: str 
    training_config: TrainingConfig


class TaskInfoResponse(BaseModel):
    task_finish_time: str 
    task_name: str 
    task_id: str
    status: str
    kwargs: dict[str, Any]
    result: Optional[Any]


class WorkerResponse(BaseModel):
    worker_address: str
    worker_port: int
    game_id: str
