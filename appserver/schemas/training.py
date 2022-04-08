from uuid import UUID
from pydantic import BaseModel
from schemas.training_parameters import TrainingParameters


class TrainingBase(BaseModel):
    full_name: str = "April first"
    endpoint_name: str = "first"


class Training(TrainingBase):
    id: UUID
    game_id: UUID
    parameters: TrainingParameters
   