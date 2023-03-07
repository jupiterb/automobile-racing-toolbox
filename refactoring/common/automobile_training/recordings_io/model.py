import numpy as np
from pydantic import BaseModel
import tables as tb
from typing import Union


class RecordingModel(BaseModel):
    """Structured representation of a recording"""

    fps: int
    frames: Union[np.ndarray, tb.Array]
    actions: Union[np.ndarray, tb.Array]

    class Config:
        arbitrary_types_allowed = True
