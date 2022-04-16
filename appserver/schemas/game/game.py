from pydantic import BaseModel
from typing import Optional

from schemas.game.game_global_configuration import GameGlobalConfiguration
from schemas.game.game_system_configuration import GameSystemConfiguration


class Game(BaseModel):
    id: str
    description: Optional[str] = None
    global_configuration: Optional[GameGlobalConfiguration] = None
    system_configuration: Optional[GameSystemConfiguration] = None
