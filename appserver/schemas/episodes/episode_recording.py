from pydantic import BaseModel
# from enviroments.real.interface.local import Frame
from schemas import State, Action
import numpy as np 

class EpisodeRecording(BaseModel):
    error: str = ""
    fps: int
    recording: list[tuple[list[list[int]], Action]] = []
