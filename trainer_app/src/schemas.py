from pydantic import BaseModel, validator, Field 
from typing import Optional, Any
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig
import uuid 
from datetime import datetime 


class TrainingRequet(BaseModel):
    training_config: TrainingConfig
    wandb_group: str = Field(default_factory=lambda: str(uuid.uuid1()))
    wandb_api_key: str 


class StartTrainingRequest(TrainingRequet):
    game_config: GameConfiguration
    env_config: EnvConfig
    wandb_run_reference: Optional[str] = None 
    checkpoint_name: Optional[str] = None 


class ResumeTrainingRequest(TrainingRequet):
    wandb_run_reference: str 
    checkpoint_name: str
    game_id: str # TODO: think about a way to extract it from wandb


class TaskInfoResponse(BaseModel):
    task_finish_time: Optional[datetime] 
    task_name: Optional[str] 
    task_id: Optional[str]
    status: Optional[str]
    result: Optional[Any]


class WorkerResponse(BaseModel):
    worker_address: str
    worker_port: int
    game_id: str
