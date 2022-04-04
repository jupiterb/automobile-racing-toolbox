from pydantic import BaseModel


class GameSystemConfiguration(BaseModel):
    path_to_geame_exe: str = ""
