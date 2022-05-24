from pydantic import BaseModel
from schemas.utils import ScreenFrame


class SegmentDetectionParams(BaseModel):
    minimal_segment_coverage: float = 0.65
    segments_definitions: list[ScreenFrame] = [
        ScreenFrame(top=0, bottom=0.2, left=0.25, right=0.75),
        ScreenFrame(top=0.2, bottom=0.4, left=0, right=0.25),
        ScreenFrame(top=0.2, bottom=0.4, left=0.75, right=1),
        ScreenFrame(top=0.4, bottom=0.6, left=0.25, right=0.75),
        ScreenFrame(top=0.6, bottom=0.8, left=0, right=0.25),
        ScreenFrame(top=0.6, bottom=0.8, left=0.75, right=1),
        ScreenFrame(top=0.8, bottom=1, left=0.25, right=0.75)
    ]
    digits_definitions: list[list[int]] = [
        [0, 1, 2, 4, 5, 6],
        [2, 5],
        [0, 2, 3, 4, 6],
        [0, 2, 3, 5, 6],
        [1, 2, 3, 5],
        [0, 1, 3, 5, 6],
        [0, 1, 3, 4, 5, 6],
        [0, 2, 5],
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 5, 6]
    ]
