from pydantic import BaseModel
from typing import Optional
from schemas.training.training_parameters import TrainingParameters
from schemas.training.training_result import TrainingResult


class Training(BaseModel):
    id: str
    description: Optional[str] = None
    parameters: TrainingParameters = TrainingParameters()
    result: TrainingResult = TrainingResult()
    __version: int = 0  # every time training is continued, version is incremented

    @property
    def version(self):
        return self.__version

    def update_result(self):
        self.__version += 1
        self.result = ...
        self.result.dump()
