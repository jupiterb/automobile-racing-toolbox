from pydantic import BaseModel
from schemas import Action
from numpy import ndarray


class EpisodeRecording(BaseModel):
    error: str = ""
    recording: list[tuple[ndarray, int, Action]] = []

    class Config:
        arbitrary_types_allowed = True
