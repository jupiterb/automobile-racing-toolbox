from pydantic import BaseModel
from schemas.training.training_parameters import TrainingParameters


class Training(BaseModel):
    id: str
    game_id: str
    description: str
    parameters: TrainingParameters
   