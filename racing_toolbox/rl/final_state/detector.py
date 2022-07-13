from typing import Optional
from rl.models import FinalFeatureValueDetectionParameters


class FinalStateDetector:
    def __init__(
        self, all_parameters: list[FinalFeatureValueDetectionParameters]
    ) -> None:
        self._all_parameters = all_parameters
        self._other_value_condition = {
            parameters.feature_name: parameters.other_value_required
            for parameters in all_parameters
        }
        self._repetitions = {
            parameters.feature_name: 0 for parameters in all_parameters
        }
                    
    def is_final(self, new_features: Optional[dict[str, float]] = None) -> bool:
        """
        Ckecks if after taking to account new features, there is final state or not.
        Parameters:
            new_features (Optional[dict[str, float]]): new features valus, None by default
        Returns:
            (bool): True if repetitions of specified value of specified feature are equal to required numbers of repetitions
                    False in other way
        """
        if new_features:
            self._consider(new_features)
        for parameters in self._all_parameters:
            name = parameters.feature_name
            if self._repetitions[name] < parameters.required_repetitions_in_row:
                return False
        return True
    
    def _consider(self, features: dict[str, float]) -> None:
        """
        Increment number of repetitions for sepcfied features if new value is equal to specfied final value 
        (and other_value_condition is passed, it means that before final value there should be at least one other value).
        If new value is not equal to specfied final value, repetitions for given feature are reset (set to 0)
        """
        for parameters in self._all_parameters:
            name = parameters.feature_name
            if name in features:
                if features[name] == parameters.final_value and not self._other_value_condition[name]:
                    self._repetitions[name] += 1
                elif features[name] != parameters.final_value:
                    if self._other_value_condition[name]:
                        self._other_value_condition[name] = False
                    self._repetitions[name] = 0
        