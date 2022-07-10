from rl.models import FinalValeDetectionConfiguration


class FinalStateDetector:
    def __init__(
        self, final_values_configuration: list[FinalValeDetectionConfiguration]
    ) -> None:
        self._final_values_configuration = final_values_configuration
        self._other_value_condition = {
            configuration.value_name: configuration.other_value_required
            for configuration in final_values_configuration
        }
        self._repetitions = {
            configuration.value_name: 0 for configuration in final_values_configuration
        }

    def consider(self, values: dict[str, float]) -> None:
        for configuration in self._final_values_configuration:
            name = configuration.value_name
            if name in values:
                if values[name] == configuration.final_value and not self._other_value_condition[name]:
                    self._repetitions[name] += 1
                elif values[name] != configuration.final_value:
                    if self._other_value_condition[name]:
                        self._other_value_condition[name] = False
                    self._repetitions[name] = 0
                    
    def get_detection(self) -> bool:
        for configuration in self._final_values_configuration:
            name = configuration.value_name
            if self._repetitions[name] != configuration.required_repetitions_in_row:
                return False
        return True
        