from numpy import character
from pydantic import BaseModel
from typing import Optional


class Action(BaseModel):
    keys: Optional[set[str]] = None
