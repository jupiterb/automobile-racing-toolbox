from uuid import UUID
from pydantic import BaseModel
from typing import List

from schemas.game_global_configuration import GameGlobalConfiguration
from schemas.game_system_configuration import GameSystemConfiguration
from schemas.training import Training

class GameEnviromentBase(BaseModel):
    full_name: str = "Trackmania Nations Forever"
    endpoint_name: str = "trackmania"


class GameEnviroment(GameEnviromentBase):
    id: UUID
    global_configuration: GameGlobalConfiguration
    system_configuration: GameSystemConfiguration
    trainings: List[Training] 