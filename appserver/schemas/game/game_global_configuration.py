from schemas.enviroment.steering import SteeringAction
from pydantic import BaseModel
from pynput.keyboard import Listener, Key, Controller


class GameGlobalConfiguration(BaseModel):
    process_name: str = ""

    action_key_mapping: dict[SteeringAction, Key] = {
        SteeringAction.FORWARD: Key.up,
        SteeringAction.LEFT: Key.left,
        SteeringAction.RIGHT: Key.right,
        SteeringAction.BREAK: Key.down,
    }
