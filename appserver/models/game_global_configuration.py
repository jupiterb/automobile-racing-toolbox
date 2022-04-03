from pydantic import BaseModel
from typing import Dict


class GameGlobalConfiguration(BaseModel):
    control_actions: Dict[str, str]
