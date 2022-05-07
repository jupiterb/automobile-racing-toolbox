from pydantic import BaseModel
from schemas import State, Action


class EpisodeRecording(BaseModel):
    error: str = ""
    fps: int
    recording: list[tuple[State, Action]] = []
