from pydantic import BaseModel


class OcrVelocityParams(BaseModel):
    binarization_threshold: int = 160
    dilate_erode_combination: list[tuple[int, list[list[int]]]] = [
        (1, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    ] # iterations + kernel. if iterations < 0 there id erode, if iterations > 0 there is dilatation
    min_width: int = 10
    min_height: int = 40
