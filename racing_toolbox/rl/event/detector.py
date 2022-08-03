from typing import Optional
from rl.config import EventDetectionParameters


class EventDetector:
    def __init__(self, all_parameters: list[EventDetectionParameters]) -> None:
        self._all_params = all_parameters
        self._other_value_condition = {
            params.feature_name: params.different_values_required
            for params in all_parameters
        }
        self._repetitions = {params.feature_name: 0 for params in all_parameters}
        if any(
            [params for params in all_parameters if params.min_value > params.max_value]
        ):
            raise ValueError("Minimal value should be lower than maximum value")
        if min(self._other_value_condition.values()) < 0:
            raise ValueError("Not event values required number should be non-negative")

    def is_final(self, new_features: Optional[dict[str, float]] = None) -> bool:
        """
        Ckecks if after taking to account new features, there is final state or not.
        Parameters:
            new_features (Optional[dict[str, float]]): new features values, None by default
        Returns:
            (bool): True if all numbers of repetitions of final values meet required numbers of repetitions
                    False in other way
        """
        if new_features:
            self._consider(new_features)
        for params in self._all_params:
            name = params.feature_name
            if self._repetitions[name] < params.required_repetitions_in_row:
                return False
        return True

    def reset(self):
        for params in self._all_params:
            name = params.feature_name
            self._other_value_condition[name] = params.different_values_required
            self._repetitions[name] = 0

    def _consider(self, features: dict[str, float]) -> None:
        """
        Increment number of repetitions for sepcfied features
        if new value is NOT in range <min_value : max_value>
        and there is at least not_event_values_required values in range <min_value : max_value>.
        If new value is in range <min_value : max_value>,
        repetitions for given feature are reset (set to 0).
        """
        for params in filter(lambda p: p.feature_name in features, self._all_params):
            name = params.feature_name
            if params.min_value <= features[name] <= params.max_value:
                if not self._other_value_condition[name]:
                    self._repetitions[name] += 1
            else:
                if self._other_value_condition[name]:
                    self._other_value_condition[name] -= 1
                self._repetitions[name] = 0
