import threading
from typing import Optional

from schemas import GameSystemConfiguration, GameGlobalConfiguration, EpisodeRecording
from enviroments import RealTimeWrapper, RealGameWrapper
from utils.custom_exceptions import WindowNotFound


class EpisodeRecordingManager():

    __default_fps: int = 30

    @staticmethod
    def default_fps():
        return EpisodeRecordingManager.__default_fps
    
    def __init__(self) -> None:
        self.__current_recording: EpisodeRecording = EpisodeRecording(
            fps=EpisodeRecordingManager.default_fps()
        )
        self.__recording_thread: Optional[threading.Thread] = None
        self.__capturing: bool = False
        self.__running: bool = True

    def __record(self, 
            system_configuration: GameSystemConfiguration,
            global_configuration: GameGlobalConfiguration,
            fps: int
        ) -> EpisodeRecording:
        enviroment_warpper = RealGameWrapper(global_configuration, system_configuration)
        self.__current_recording = EpisodeRecording(fps=fps)
        try:
            while self.__running:
                if self.__capturing:
                    state = enviroment_warpper.read_state()
                    action = enviroment_warpper.read_action()
                    self.__current_recording.recording.append((state, action))
        except WindowNotFound as e:
            self.__current_recording.error=f"Process {e.process_name} do not exist"
        return self.__current_recording

    def start(self, 
            system_configuration: GameSystemConfiguration,
            global_configuration: GameGlobalConfiguration,
            fps: int
        ):
        self.__recording_thread = threading.Thread(
                target=self.__record,
                args=(system_configuration, global_configuration, fps)
            )
        self.__capturing = True
        self.__running = True
        self.__recording_thread.start()

    def stop(self):
        self.__capturing = False

    def resume(self):
        self.__capturing = True

    def release(self) -> EpisodeRecording:
        self.__capturing = False
        self.__running = False
        if self.__recording_thread is not None:
            self.__recording_thread.join()
        return self.__current_recording
        