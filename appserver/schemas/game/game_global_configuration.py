from pydantic import BaseModel

from .feature_extraction import OcrVelocityParams


class GameGlobalConfiguration(BaseModel):
    process_name: str = ""
    control_actions: set[str] = set(['right', 'left', 'down', 'up'])
    ocr_velocity_params: OcrVelocityParams = OcrVelocityParams()
    