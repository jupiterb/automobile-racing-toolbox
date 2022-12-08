from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ConfigSourceType(Enum):
    FILE_BASED = 0
    S3_BUCKET = 1


class Credentials(BaseModel):
    user_key_id: str
    user_secret_key: str


class ConfigsContentEndpoints(BaseModel):
    game_configs: str
    env_configs: str
    training_configs: str
    recordings: str


class UIAppConfig(BaseModel):
    source_type: ConfigSourceType
    credentials: Optional[Credentials]
    root: str
    endpoints: ConfigsContentEndpoints
