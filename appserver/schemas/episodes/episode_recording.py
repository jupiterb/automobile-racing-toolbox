from pydantic import BaseModel
from enviroments.real.interface.local import Frame
from schemas import State, Action


class EpisodeRecording(BaseModel):
    error: str = ""
    fps: int
    recording: list[tuple[Frame, Action]] = []
