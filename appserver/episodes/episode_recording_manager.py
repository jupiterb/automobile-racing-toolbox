from schemas import GameSystemConfiguration, GameGlobalConfiguration, EpisodeRecording
from enviroments import RealTimeWrapper, RealGameWrapper
from utils.custom_exceptions import WindowNotFound


class EpisodeRecordingManager():

    def record(self, 
            system_configuration: GameSystemConfiguration,
            global_configuration: GameGlobalConfiguration
        ) -> EpisodeRecording:
        enviroment_warpper = RealGameWrapper(global_configuration, system_configuration)
        try:
            state = enviroment_warpper.read_state()
        except WindowNotFound as e:
            return EpisodeRecording(error=f"Process {e.process_name} do not exist")
        return EpisodeRecording()
