from enum import Enum 

class KeyAction(str, Enum):
    up = "up"
    down = "down"
    left = "left"
    right = "right"
    space = "space"
    enter = "enter"
    