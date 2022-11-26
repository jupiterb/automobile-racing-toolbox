from vgamepad import XUSB_BUTTON
from enum import Enum
from typing import Union


max_xusb = max(XUSB_BUTTON)


class GamepadControl(Enum):
    AXIS_X_LEFT = max_xusb + 1
    AXIS_Y_LEFT = max_xusb + 2
    AXIS_X_RIGHT = max_xusb + 3
    AXIS_Y_RIGHT = max_xusb + 4
    AXIS_Z = max_xusb + 5


GamepadAction = Union[GamepadControl, XUSB_BUTTON]
