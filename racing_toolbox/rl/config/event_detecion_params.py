from pydantic import BaseModel


class EventDetectionParameters(BaseModel):
    feature_name: str
    min_value: float
    max_value: float
    required_repetitions_in_row: int
    different_values_required: int
    