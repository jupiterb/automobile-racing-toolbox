import threading
from typing import Optional
import time

from interface import GameInterface
from recorderapp.dataservice import RecorderDataService, InMemoryDataService

from rl.final_state import FinalStateDetector
from rl.models import FinalValeDetectionConfiguration


class EpisodeRecordingManager:

    __default_fps: int = 10

    @staticmethod
    def default_fps():
        return EpisodeRecordingManager.__default_fps

    def __init__(self) -> None:
        self.__dataservice: RecorderDataService = InMemoryDataService()
        self.__recording_thread: Optional[threading.Thread] = None
        self.__capturing: bool = False
        self.__running: bool = True

    def start(
        self,
        interface: GameInterface,
        user: str,
        recording_name: str,
        fps: int,
    ):
        self.__dataservice.start_streaming(interface.name(), user, recording_name, fps)
        self.__recording_thread = threading.Thread(
            target=self.__record, args=(interface, fps)
        )
        self.__capturing = True
        self.__running = True
        self.__recording_thread.start()

    def stop(self):
        self.__capturing = False

    def resume(self):
        self.__capturing = True

    def release(self) -> None:
        self.__capturing = False
        self.__running = False
        if self.__recording_thread is not None:
            self.__recording_thread.join()
        self.__dataservice.stop_streaming()

    def running(self) -> bool:
        return self.__running

    def caturing(self) -> bool:
        return self.__capturing

    def __record(
        self,
        game_interface: GameInterface,
        fps: int,
    ) -> None:
        _ = game_interface.reset()
        while self.__running:
            if self.__capturing:
                image = game_interface.grab_image()
                from_ocr = game_interface.perform_ocr()
                action = game_interface.read_action()
                self.__dataservice.put_observation(image, from_ocr, set(action))
                time.sleep(1 / fps)
