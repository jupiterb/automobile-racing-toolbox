from pydantic import BaseModel


class GameGlobalConfiguration(BaseModel):
    control_actions: dict[str, str] = {}
