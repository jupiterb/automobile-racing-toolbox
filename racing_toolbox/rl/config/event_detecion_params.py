from pydantic import BaseModel
from typing import Optional


class EventDetectionParameters(BaseModel):
    feature_name: str
    min_value: Optional[float]
    max_value: Optional[float]
    required_repetitions_in_row: int
    not_event_values_required: int
    