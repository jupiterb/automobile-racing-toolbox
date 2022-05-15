from pydantic import BaseModel
from schemas.utils import ScreenFrame


class SegmentDetectionParams(BaseModel):
    minimal_segment_coverage: float = 0.65
    segments_definitions: dict[int, ScreenFrame] = {
        0: ScreenFrame(top=0, bottom=0.2, left=0.25, right=0.75),
        1: ScreenFrame(top=0.2, bottom=0.4, left=0, right=0.25),
        2: ScreenFrame(top=0.2, bottom=0.4, left=0.75, right=1),
        3: ScreenFrame(top=0.4, bottom=0.6, left=0.25, right=0.75),
        4: ScreenFrame(top=0.6, bottom=0.8, left=0, right=0.25),
        5: ScreenFrame(top=0.6, bottom=0.8, left=0.75, right=1),
        6: ScreenFrame(top=0.8, bottom=1, left=0.25, right=0.75)
    }
    digits_definitions: dict[int, list[int]] = {
        0: [0, 1, 2, 4, 5, 6],
        1: [2, 5],
        2: [0, 2, 3, 4, 6],
        3: [0, 2, 3, 5, 6],
        4: [1, 2, 3, 5],
        5: [0, 1, 3, 5, 6],
        6: [0, 1, 3, 4, 5, 6],
        7: [0, 2, 5],
        8: [0, 1, 2, 3, 4, 5, 6],
        9: [0, 1, 2, 3, 5, 6]
    }
