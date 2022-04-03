from uuid import UUID
from pydantic import BaseModel
from typing import Optional


class Agent(BaseModel):
    id: UUID
    name: str = "Kubica"
