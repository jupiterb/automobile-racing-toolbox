from schemas import GameSystemConfiguration, GameGlobalConfiguration, TrainingParameters


class TrainingManager(object):

    def run_training(slef, 
            system_configuratio: GameSystemConfiguration,
            global_configuration: GameGlobalConfiguration, 
            training_parameters: TrainingParameters
        ):
        pass

    def stop_training(self):
        pass
