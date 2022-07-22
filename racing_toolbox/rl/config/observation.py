from pydantic import BaseModel, PositiveInt


class ObservationConfig(BaseModel):
    shape: tuple[PositiveInt, PositiveInt] 
    stack_size: PositiveInt


    
