from pydantic import BaseModel


class OcrConfiguration(BaseModel):
    threshold: int
    segments_coordinates: dict[int, tuple[float, float]] | None
