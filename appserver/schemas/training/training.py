from pydantic import BaseModel
from typing import Optional
from schemas.training.training_parameters import TrainingParameters


class Training(BaseModel):
    id: str
    description: Optional[str] = None
    parameters: Optional[TrainingParameters] = None
   