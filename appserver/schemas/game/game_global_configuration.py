from pydantic import BaseModel


class GameGlobalConfiguration(BaseModel):
    process_name: str = ""
    control_actions: set[str] = set(['right', 'left', 'down', 'up'])
