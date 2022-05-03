from pydantic import BaseModel
from typing import Literal, Optional


class Action(BaseModel):
    key: Optional[Literal] = None
