from pydantic import BaseModel


class ScreenFrame(BaseModel):
    top: float = 0.0
    bottom: float = 1.0
    left: float = 0.0
    right: float = 1.0
    