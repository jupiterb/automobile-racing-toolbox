from pydantic import BaseModel
from typing import Optional
from enum import Enum, auto

from schemas.game.feature_extraction.segment_detection_params import SegmentDetectionParams


class OcrType(Enum):
    SEGMENT_DETECTION = auto()


class OcrVelocityParams(BaseModel):
    dilate_erode_combination: list[tuple[int, list[list[int]]]] = [
        (2, [[1, 1,], [1, 1], [1, 1]])
    ] # iterations + kernel. if iterations < 0 there id erode, if iterations > 0 there is dilatation
    min_width: int = 10
    min_height: int = 40
    shape_width: int = 40
    shape_height: int = 50
    ocr_type: OcrType = OcrType.SEGMENT_DETECTION
    segment_detection_params: Optional[SegmentDetectionParams] = SegmentDetectionParams()
