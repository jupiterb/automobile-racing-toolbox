from pydantic import BaseModel
from enum import Enum, auto


class NoResultPolicy(Enum):
    RETURN_ZERO = auto()
    RETURN_LAST = auto()
    RETURN_NEGATIVE = auto()


class OcrConfiguration(BaseModel):
    threshold: int
    segments_coordinates: dict[int, tuple[float, float]] | None
    unknown_elements_policy: NoResultPolicy
    lack_of_elements_policy: NoResultPolicy
    try_lower_threshold: bool
