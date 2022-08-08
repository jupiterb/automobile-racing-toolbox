from pydantic import BaseModel


class TrackSegmentationConfig(BaseModel):
    lower_threshold: int
    upper_threshold: int
    kernel_size: int
