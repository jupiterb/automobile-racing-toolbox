from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum, auto

from schemas.game.feature_extraction.segment_detection_params import (
    SegmentDetectionParams,
)


class OcrType(Enum):
    SEGMENT_DETECTION = auto()


class MorphologyOperationType(Enum):
    DILATING = auto()
    EROSION = auto()


class MorphologyOperation(BaseModel):
    type: MorphologyOperationType
    iterations: int
    kernel: list[list[int]]


class OcrVelocityParams(BaseModel):
    morphology_operations_combination: list[MorphologyOperation] = [
        MorphologyOperation(
            type=MorphologyOperationType.DILATING,
            iterations=1,
            kernel=[[1, 1], [1, 1], [1, 1]],
        )
    ]
    absolute_min_width: float = 0.075
    absolute_min_height: float = 0.75
    min_width: float = 0.15
    shape_width: int = 40
    shape_height: int = 50
    ocr_type: OcrType = OcrType.SEGMENT_DETECTION
    segment_detection_params: Optional[
        SegmentDetectionParams
    ] = SegmentDetectionParams()
