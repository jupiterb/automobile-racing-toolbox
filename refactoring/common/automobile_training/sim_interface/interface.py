import numpy as np
import time

from automobile_training.sim_interface.vision_capturing import VisionCapturing
from automobile_training.sim_interface.action_controller import ActionController
from automobile_training.sim_interface.action_capturing import ActionCapturing


class BaseVisionOnlySimInterface:
    """Base class for ReactiveSimInterface and CaptureSimInterface, both have VisionCpaturing"""

    def __init__(self, sim_vision: VisionCapturing) -> None:
        self._sim_vision = sim_vision

    def get_state(self) -> np.ndarray:
        return self._sim_vision.get_vision()


class InteractiveSimInterface(BaseVisionOnlySimInterface):
    """Type of interface used by agent to interact with simulation"""

    def __init__(
        self,
        sim_vision: VisionCapturing,
        controller: ActionController,
        reset_delay: int,
    ) -> None:
        super().__init__(sim_vision)
        self._controller = controller
        self._reset_delay = reset_delay

    @property
    def possible_actions(self) -> set[str]:
        return self._controller.possible_actions

    def apply_actions(self, actions: dict[str, float]):
        self._controller.apply(actions)

    def reset(self):
        self._controller.reset()
        time.sleep(self._reset_delay)


class CaptureSimInterface(BaseVisionOnlySimInterface):
    """Type of interface used in recording of expert behavior"""

    def __init__(
        self, sim_vision: VisionCapturing, actions_capturing: ActionCapturing
    ) -> None:
        super().__init__(sim_vision)
        self._actions_capturing = actions_capturing

    def get_actions(self) -> dict[str, float]:
        return self._actions_capturing.get_actions()

    def set_capturing(self, capture: bool):
        if capture:
            self._actions_capturing.start()
        else:
            self._actions_capturing.stop()
