from schemas.enviroment.steering import SteeringAction
from pydantic import BaseModel
from pynput.keyboard import Key

from .feature_extraction import OcrVelocityParams


class GameGlobalConfiguration(BaseModel):
    process_name: str = ""
    action_key_mapping: dict[SteeringAction, str] = {
        SteeringAction.FORWARD: Key.up.name,
        SteeringAction.LEFT: Key.left.name,
        SteeringAction.RIGHT: Key.right.name,
        SteeringAction.BREAK: Key.down.name,
    }
    # TODO: Add validator to make sure all SteeringAction were translated to the keys
    ocr_velocity_params: OcrVelocityParams = OcrVelocityParams()
    observation_shape: tuple[int, int] = (100, 100)
