from uuid import UUID
from pydantic import BaseModel


class TrainingBase(BaseModel):
    full_name: str = "April first"
    endpoint_name: str = "first"


class TrainingParameters(BaseModel):
    any_parameter: int


class Training(TrainingBase. TrainingParameters):
    id: UUID
    game_id: UUID
   