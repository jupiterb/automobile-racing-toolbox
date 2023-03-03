import numpy as np
import time

from automobile_training.sim_interface.vision_capturing import VisionCapturing
from automobile_training.sim_interface.input_controller import InputController
from automobile_training.sim_interface.input_capturing import InputCapturing


class BaseVisionOnlySimInterface:
    """Base class for ReactiveSimInterface and CaptureSimInterface, both have VisionCpaturing"""

    def __init__(self, sim_vision: VisionCapturing) -> None:
        self._sim_vision = sim_vision

    def get_state(self) -> np.ndarray:
        return self._sim_vision.get_vision()


class InteractiveSimInterface(BaseVisionOnlySimInterface):
    """Type of interface used by agent to interact with simulation"""

    def __init__(
        self, sim_vision: VisionCapturing, controller: InputController, reset_delay: int
    ) -> None:
        super().__init__(sim_vision)
        self._controller = controller
        self._reset_delay = reset_delay

    @property
    def possible_inputs(self) -> set[str]:
        return self._controller.possible_inputs

    def apply_inputs(self, inputs: dict[str, float]):
        self._controller.apply(inputs)

    def reset(self):
        self._controller.reset()
        time.sleep(self._reset_delay)


class CaptureSimInterface(BaseVisionOnlySimInterface):
    """Type of interface used in recording of expert behavior"""

    def __init__(
        self, sim_vision: VisionCapturing, inputs_capturing: InputCapturing
    ) -> None:
        super().__init__(sim_vision)
        self._inputs_capturing = inputs_capturing

    def get_inputs(self) -> dict[str, float]:
        return self._inputs_capturing.get_inputs()

    def set_capturing(self, capture: bool):
        if capture:
            self._inputs_capturing.start()
        else:
            self._inputs_capturing.stop()
