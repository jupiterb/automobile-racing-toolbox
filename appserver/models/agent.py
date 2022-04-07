from uuid import UUID
from pydantic import BaseModel


class Agent(BaseModel):
    id: UUID
    game_id: UUID
    full_name: str = "Kubica"
    endpoint_name: str = "kubica"
