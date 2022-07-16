from typing import Optional
import numpy as np

from rl.models import FinalValueDetectionParameters


class FinalStateDetector:
    def __init__(self, all_parameters: list[FinalValueDetectionParameters]) -> None:
        self._all_parameters = all_parameters
        self._other_value_condition = {
            parameters.feature_name: parameters.not_final_value_required
            for parameters in all_parameters
        }
        self._repetitions = {
            parameters.feature_name: 0 for parameters in all_parameters
        }
        self._ranges = {
            parameters.feature_name: (
                -np.inf if parameters.min_value is None else parameters.min_value,
                np.inf if parameters.max_value is None else parameters.max_value,
            )
            for parameters in all_parameters
        }
        for min_value, max_value in self._ranges.values():
            if min_value > max_value:
                raise ValueError("Minimal value should be lower than maximum value")

    def is_final(self, new_features: Optional[dict[str, float]] = None) -> bool:
        """
        Ckecks if after taking to account new features, there is final state or not.
        Parameters:
            new_features (Optional[dict[str, float]]): new features values, None by default
        Returns:
            (bool): True if any number of repetitions of final values meet required numbers of repetitions
                    False in other way
        """
        if new_features:
            self._consider(new_features)
        for parameters in self._all_parameters:
            name = parameters.feature_name
            if self._repetitions[name] >= parameters.required_repetitions_in_row:
                return True
        return False

    def reset(self):
        for parameters in self._all_parameters:
            name = parameters.feature_name
            self._other_value_condition[name] = False
            self._repetitions[name] = 0


    def _consider(self, features: dict[str, float]) -> None:
        """
        Increment number of repetitions for sepcfied features if new value is NOT in range <min_value : max_value>
        (and not_final_value_condition is passed, it means that before final value there should be at least one other value).
        If new value is in range <min_value : max_value>, repetitions for given feature are reset (set to 0).
        min_value = None means -inf, while max_value = None means +inf.
        """
        for parameters in self._all_parameters:
            name = parameters.feature_name
            if name in features:
                min_value, max_value = self._ranges[name]
                if not min_value <= features[name] <= max_value:
                    if not self._other_value_condition[name]:
                        self._repetitions[name] += 1
                else:
                    self._other_value_condition[name] = False
                    self._repetitions[name] = 0
