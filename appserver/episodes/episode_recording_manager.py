from schemas import GameSystemConfiguration, GameGlobalConfiguration, EpisodeRecording
from enviroments import RealTimeWrapper, RealGameWrapper


class EpisodeRecordingManager():

    def record(self, 
            system_configuration: GameSystemConfiguration,
            global_configuration: GameGlobalConfiguration
        ) -> EpisodeRecording:
        enviroment_warpper = RealGameWrapper(global_configuration, system_configuration)
        state = enviroment_warpper.read_state()
        return EpisodeRecording()
