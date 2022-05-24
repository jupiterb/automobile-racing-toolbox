from pydantic import BaseModel
from schemas import Action
from typing import Union


ImageMatrix = Union[list[list[int]], list[list[list[int]]]]


class EpisodeRecording(BaseModel):
    error: str = ""
    fps: int
    recording: list[tuple[ImageMatrix, Action]] = []
