from schemas.enviroment.steering import SteeringAction
from pydantic import BaseModel
from pynput.keyboard import Key

from .feature_extraction import OcrVelocityParams


class GameGlobalConfiguration(BaseModel):
    process_name: str = ""
    action_key_mapping: dict[SteeringAction, Key] = {
        SteeringAction.FORWARD: Key.up,
        SteeringAction.LEFT: Key.left,
        SteeringAction.RIGHT: Key.right,
        SteeringAction.BREAK: Key.down,
    }
    # TODO: Add validator to make sure all SteeringAction were translated to the keys
    ocr_velocity_params: OcrVelocityParams = OcrVelocityParams()
    observation_shape: tuple[int, int] | tuple[int, int, int] = (100, 100)
