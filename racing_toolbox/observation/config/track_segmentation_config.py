from pydantic import BaseModel


class TrackSegmentationConfig(BaseModel):
    track_color: tuple[int, int, int]
    tolerance: int
    noise_reduction: int
