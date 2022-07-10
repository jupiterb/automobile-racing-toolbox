from argparse import ArgumentError
import numpy as np
from typing import Optional
from observation.models import ObservationConfiguration


class ObservationCreator:
    def __init__(
        self,
        configuration: ObservationConfiguration,
        image_shape: tuple[int, int],
        values_number: int,
    ) -> None:
        self._configuration = configuration
        self._values_buffers = [
            np.zeros(
                shape=(configuration.buffer_size + 1, image_shape[0], image_shape[1])
            )
            for _ in range(configuration.offset)
        ]
        self._images_buffers = [
            np.zeros(shape=(configuration.buffer_size + 1, values_number))
            for _ in range(configuration.offset)
        ]
        self._buffer_occupancy = 0
        self._buffer_index = 0

    def add_to_buffer(self, image: np.ndarray, values: np.ndarray) -> None:
        if image.ndim != 2 or image.ndim != 3:
            raise ValueError("Input image should be 2 or 3-dimensional")
        if values.ndim != 1:
            raise ValueError("Input vales should be 1-dimensional")

        images_buffer = self._images_buffers[self._buffer_index]
        values_buffer = self._values_buffers[self._buffer_index]

        images_buffer[1 : self._buffer_occupancy + 1] = images_buffer[
            0 : self._buffer_occupancy
        ]
        values_buffer[1 : self._buffer_occupancy + 1] = values_buffer[
            0 : self._buffer_occupancy
        ]

        images_buffer[0] = image
        values_buffer[0] = values

        self._buffer_index = (self._buffer_index + 1) % self._configuration.offset
        self._buffer_occupancy = max(
            self._configuration.buffer_size, self._buffer_occupancy + 1
        )

    def get_observation(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        buffer_size = self._configuration.buffer_size
        if self._buffer_occupancy != buffer_size:
            return None
        return (
            self._images_buffers[self._buffer_index][0:buffer_size],
            self._values_buffers[self._buffer_index][0:buffer_size],
        )
