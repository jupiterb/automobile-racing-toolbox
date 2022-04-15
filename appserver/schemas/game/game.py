from pydantic import BaseModel
from typing import List, Optional

from schemas.game.game_global_configuration import GameGlobalConfiguration
from schemas.game.game_system_configuration import GameSystemConfiguration
from schemas.training import Training


class Game(BaseModel):
    id: str
    description: str
    global_configuration: Optional[GameGlobalConfiguration] = None
    system_configuration: Optional[GameSystemConfiguration] = None
    trainings: List[Training] = []
