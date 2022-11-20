from typing import Optional
from pydantic import BaseModel


class ActionConfig(BaseModel):
    # If set, DiscreteActionToVectorWrapper may be used
    available_actions: Optional[dict[str, set[int]]]
