from schemas import GameSystemConfiguration, GameGlobalConfiguration, TrainingParameters, TrainingResult


class TrainingManager():

    def run_training(self, 
            system_configuration: GameSystemConfiguration,
            global_configuration: GameGlobalConfiguration,
            training_parameters: TrainingParameters
        ):
        pass

    def stop_training(self) -> TrainingResult:
        return TrainingResult()
