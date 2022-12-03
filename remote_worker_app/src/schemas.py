from pydantic import BaseModel, BaseSettings, validator
from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from remote_worker_app.src.worker import Worker, Address
import socket


class SyncRequest(BaseModel):
    game_config: GameConfiguration
    env_config: EnvConfig
    policy_address: Address
    wandb_project: str
    wandb_api_key: str
    wandb_group: str


import uuid
from typing import Optional


class EnvVars(BaseSettings):
    host: str
    port: int
    register_url: str
    keepalive_url: str
    game_id: str
    self_id: Optional[uuid.UUID] = None

    @validator("host")
    def change_to_local_address(cls, v):
        if v != "0.0.0.0":
            return v
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address

    class Config:
        env_file = ".env"

    @property
    def self_url(self) -> str:
        return f"http://{self.host}:{self.port}"
