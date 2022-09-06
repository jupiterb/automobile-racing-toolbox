from pydantic import BaseModel


class SteeringAction(BaseModel):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    BREAK = 3
