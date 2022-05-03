from pydantic import BaseModel
from typing import Optional
from numpy import ndarray


class State(BaseModel):
    screenshot: Optional[ndarray] = None
    velocity: Optional[int] = None
