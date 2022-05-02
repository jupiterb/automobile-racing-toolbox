from pydantic import BaseModel


class EpisodeRecording(BaseModel):
    error: str = ""
