import threading
from typing import Optional, Callable
import time

from racing_toolbox.interface import GameInterface
from racing_toolbox.datatool.services import (
    InMemoryDatasetService,
    AbstractDatasetService,
)


class EpisodeRecordingManager:

    __get_dataservice: Callable[[], AbstractDatasetService]

    def __init__(self) -> None:
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
        self.__get_dataservice = lambda: InMemoryDatasetService(
            "./recordings", interface.name(), user, recording_name, fps
        )
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

    def running(self) -> bool:
        return self.__running

    def caturing(self) -> bool:
        return self.__capturing

    def __record(
        self,
        game_interface: GameInterface,
        fps: int,
    ) -> None:
        game_interface.reset()
        with self.__get_dataservice() as service:
            while self.__running:
                if self.__capturing:
                    image = game_interface.grab_image()
                    action = game_interface.read_action()
                    service.put(image, action)
                    time.sleep(1 / fps)
        game_interface.reset(False)
