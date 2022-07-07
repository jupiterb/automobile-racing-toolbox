import threading
from typing import Optional
import time

from interface import LocalGameInterface
from interface.models import GameConfiguration

from recorderapp.dataservice import RecorderDataService, InMemoryDataService


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
        configuration: GameConfiguration,
        user: str,
        recording_name: str,
        fps: int,
    ):
        self.__dataservice.start_streaming(
            configuration.game_id, user, recording_name, fps
        )
        self.__recording_thread = threading.Thread(
            target=self.__record, args=(configuration, fps)
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
        configuration: GameConfiguration,
        fps: int,
    ) -> None:
        enviroment_interface = LocalGameInterface(configuration)
        _ = enviroment_interface.restart()
        while self.__running:
            if self.__capturing:
                image = enviroment_interface.grab_image()
                from_ocr = [
                    float(value) for value in enviroment_interface.perform_ocr()
                ]
                action = enviroment_interface.read_action()
                print(image.shape, from_ocr, action)
                self.__dataservice.put_observation(image, from_ocr, set(action))
                time.sleep(1 / fps)
