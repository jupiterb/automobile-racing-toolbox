from enviroments.real.action.steering import SteeringAction
from pydantic import BaseModel
from pynput.keyboard import Listener, Key, Controller


class GameGlobalConfiguration(BaseModel):
    process_name: str = ""
    action_key_mapping: dict[SteeringAction, Key] = {
        SteeringAction.FORWARD: Key.UP,
        SteeringAction.LEFT: Key.LEFT,
        SteeringAction.RIGHT: Key.RIGHT,
        SteeringAction.STOP: Key.DOWN,
    }
    # TODO: Add validator to make sure all SteeringAction were translated to the keys
