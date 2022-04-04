from uuid import UUID
from pydantic import BaseModel
from typing import Optional


class Agent(BaseModel):
    id: UUID
    game_id: UUID
    name: str = "Kubica"
