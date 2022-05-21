from pydantic import BaseModel
from schemas import  Action


ImageMatrix = list[list[int]]


class EpisodeRecording(BaseModel):
    error: str = ""
    fps: int
    recording: list[tuple[ImageMatrix, Action]] = []
