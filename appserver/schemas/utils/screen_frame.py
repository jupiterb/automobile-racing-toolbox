from pydantic import BaseModel


class ScreenFrame(BaseModel):
    top: float
    bottom: float
    right: float
    left: float
