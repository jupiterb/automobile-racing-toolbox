from pydantic import BaseModel
from schemas import  Action, State


class EpisodeRecording(BaseModel):
    error: str = ""
    fps: int
    recording: list[tuple[State, Action]] = []
