from pydantic import BaseModel
from typing import Optional, Any
import numpy as np


class State(BaseModel):
    screenshot_numpy_array: Optional[np.ndarray] = None
    screenshot_python_array: Optional[Any] = None 
    velocity: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
