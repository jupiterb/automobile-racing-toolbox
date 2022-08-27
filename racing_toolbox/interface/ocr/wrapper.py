import numpy as np
from interface.screen import ScreenProvider
from interface.models import ScreenFrame
from interface.ocr.abstract import AbstractOcr


class OcrWrapper:
    def __init__(self, frame: ScreenFrame, name: str, wrapped_ocr: AbstractOcr) -> None:
        self._frame = frame
        self._name = name
        self._ocr = wrapped_ocr

    def name(self) -> str:
        return self._name

    def read_numer_from(self, screen: ScreenProvider, on_last: bool) -> int:
        return self._ocr.read_numer(screen.grab_image(self._frame, on_last=on_last))
