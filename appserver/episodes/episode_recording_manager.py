from schemas import GameSystemConfiguration, GameGlobalConfiguration, EpisodeRecording


class EpisodeRecordingManager():

    def start_recording(self, 
            system_configuration: GameSystemConfiguration,
            global_configuration: GameGlobalConfiguration
        ):
        pass

    def stop_recording(self) -> EpisodeRecording:
        return EpisodeRecording()
