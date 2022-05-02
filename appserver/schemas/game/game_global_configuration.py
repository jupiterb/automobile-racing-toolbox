from pydantic import BaseModel


class GameGlobalConfiguration(BaseModel):
    process_name: str = ""
    control_actions: dict[str, str] = {}
