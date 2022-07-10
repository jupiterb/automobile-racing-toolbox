from pydantic import BaseModel


class FinalValeDetectionConfiguration(BaseModel):
    value_name: str
    final_value: float
    required_repetitions_in_row: int
    other_value_required: bool
    