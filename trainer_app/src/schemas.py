from pydantic import BaseModel, validator, Field 
from typing import Optional, Any, TypeVar, overload
from racing_toolbox.training.config.user_defined import ModelConfig
from racing_toolbox.observation.config.vae_config import VAETrainingConfig, VAEModelConfig
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.observation.config import VAEConfig
from racing_toolbox.environment.config.reward import RewardConfig
from racing_toolbox.training.config.user_defined import TrainingConfig
from racing_toolbox.observation.utils.ocr import OcrToolConfiguration
import logging 
import uuid 
from datetime import datetime 

logger = logging.getLogger(__name__)

class TrainingRequet(BaseModel):
    training_config: TrainingConfig
    wandb_group: str = Field(default_factory=lambda: str(uuid.uuid1()))
    wandb_api_key: str 


class StartTrainingRequest(TrainingRequet):
    game_config: GameConfiguration
    env_config: EnvConfig
    wandb_run_reference: Optional[str] = None 
    checkpoint_name: Optional[str] = None 


_ConfType = TypeVar("_ConfType", EnvConfig, GameConfiguration)
class OverwritingConfig(BaseModel):
    # env config
    vae_config: Optional[VAEConfig]=None
    reward_config: Optional[RewardConfig]=None
    video_freq: Optional[int]=None
    video_len: Optional[int]=None

    # game config
    ocrs: Optional[OcrToolConfiguration]=None

    def maybe_overwrite(self, config: _ConfType) -> _ConfType:
        logger.warning(f"Going to overwrite {type(config)} with {self.dict(exclude_none=True)}")
        if isinstance(config, GameConfiguration):
            if self.ocrs:
                config.ocrs = self.ocrs 
        elif isinstance(config, EnvConfig):
            if self.vae_config:
                config.observation_config.vae_config = self.vae_config
            if self.reward_config:
                config.reward_config = self.reward_config
            if self.video_freq:
                config.video_freq = self.video_freq
            if self.video_len:
                config.video_len = self.video_len
        return config 


class ResumeTrainingRequest(TrainingRequet):
    overwriting_config: OverwritingConfig=Field(default_factory=OverwritingConfig)
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


class StartVAETrainingRequest(BaseModel):
    wandb_api_key: str 
    training_params: VAETrainingConfig
    encoder_config: VAEModelConfig
    bucket_name: str 
    recordings_refs: list[str]
    