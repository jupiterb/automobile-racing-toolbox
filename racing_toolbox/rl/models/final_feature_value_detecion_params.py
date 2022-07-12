from pydantic import BaseModel


class FinalFeatureValueDetectionParameters(BaseModel):
    feature_name: str
    final_value: float
    required_repetitions_in_row: int
    other_value_required: bool
    