from pydantic import BaseModel
from typing import Optional
from pynput.keyboard import Key


class Action(BaseModel):
    keys: Optional[set[Key]] = None
