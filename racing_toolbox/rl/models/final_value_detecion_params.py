from pydantic import BaseModel
from typing import Optional


class FinalValueDetectionParameters(BaseModel):
    feature_name: str
    min_value: Optional[float]
    max_value: Optional[float]
    required_repetitions_in_row: int
    not_final_value_required: bool
    