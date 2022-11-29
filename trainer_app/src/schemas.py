from pydantic import BaseModel
from typing import Optional, Any
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig


class StartTaskRequest(BaseModel):
    game_config: GameConfiguration
    env_config: EnvConfig
    training_config: TrainingConfig
    checkpoint_reference: Optional[str] = None
    model_reference: Optional[str] = None


class TaskInfoResponse(BaseModel):
    task_id: str
    status: str
    kwargs: dict[str, Any]
    result: Optional[Any]


class WorkerResponse(BaseModel):
    worker_address: str
    worker_port: int
    game_id: str
