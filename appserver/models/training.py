from uuid import UUID
from pydantic import BaseModel


class Training(BaseModel):
    id: UUID
    game_id: UUID
    full_name: str = "Any Training"
    endpoint_name: str = "any_training"
