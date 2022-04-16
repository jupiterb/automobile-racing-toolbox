from schemas import GameSystemConfiguration, GameGlobalConfiguration, TrainingParameters, TrainingResult


class TrainingManager():

    def run_training(slef, 
            system_configuration: GameSystemConfiguration,
            global_configuration: GameGlobalConfiguration,
            training_parameters: TrainingParameters
        ):
        pass

    def stop_training(self) -> TrainingResult:
        pass
