from pydantic import BaseModel


class TrainingParameters(BaseModel):
    any_parameter: int = 5
    