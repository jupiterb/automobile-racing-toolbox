from pydantic import BaseModel
from typing import Optional

from schemas.episodes.episode_recording import EpisodeRecording


class Episode(BaseModel):
    id: str
    description: Optional[str] = None
    recording: Optional[EpisodeRecording] = None
