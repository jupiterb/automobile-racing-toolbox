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
    observation_shape: tuple[int, int, int] = (100, 100, 1)
    window_size: tuple[int, int] = (1000, 800)
    apply_grayscale: bool = True

    class Config:
        json_encoders = {Key: lambda key: key.name}
