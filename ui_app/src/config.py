from pydantic import BaseModel
from typing import Optional


class UserData(BaseModel):
    email: str
    username: str
    password: str
    user_key_id: str
    user_secret_key: str
    wandb_api_key: str


class SourcesKeys(BaseModel):
    game_configs: str
    env_configs: str
    training_configs: str
    vae_training_configs: str
    vae_model_configs: str
    recordings: str


class AppConfig(BaseModel):
    trainer_url: str
    registry_url: str
    bucket_name: str
    sources_keys: SourcesKeys
