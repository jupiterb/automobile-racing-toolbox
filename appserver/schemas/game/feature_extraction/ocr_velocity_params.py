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
            iterations=2,
            kernel=[[1, 1], [1, 1], [1, 1]],
        )
    ]
    min_width: int = 10
    min_height: int = 40
    shape_width: int = 40
    shape_height: int = 50
    ocr_type: OcrType = OcrType.SEGMENT_DETECTION
    segment_detection_params: Optional[
        SegmentDetectionParams
    ] = SegmentDetectionParams()
