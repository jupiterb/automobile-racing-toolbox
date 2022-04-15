from pydantic import BaseModel
from typing import List

from schemas.game.game_global_configuration import GameGlobalConfiguration
from schemas.game.game_system_configuration import GameSystemConfiguration
from schemas.training import Training


class Game(BaseModel):
    id: str
    description: str
    global_configuration: GameGlobalConfiguration
    system_configuration: GameSystemConfiguration
    trainings: List[Training] 
