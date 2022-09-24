import pytest
import random
import shutil
import numpy as np

from racing_toolbox.datatool import RecordingsContainer
from racing_toolbox.datatool.models import Recording
from racing_toolbox.datatool.recordings import (
    BinaryFileRecordingsService,
    RecorderDataService,
)


@pytest.fixture
def my_service() -> RecorderDataService:
    path = "./temp"
    try:
        # since it's temp, items inside are no longer needed
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    return BinaryFileRecordingsService("./temp", size_limit=200)


@pytest.fixture
def observations() -> list[tuple[np.ndarray, dict[str, float], dict[str, float]]]:
    obs_number = lambda i: i % 5 + i % 3 + 1
    image = lambda i: np.full((50, 50, 3), obs_number(i))
    features = lambda i: {"a": obs_number(i), "b": -obs_number(i)}
    actions = lambda i: {"x": obs_number(i) / 2, "y": -obs_number(i) / 4}
    return [(image(i), features(i), actions(i)) for i in range(100)]


@pytest.fixture
def shuffled_observations(
    observations,
) -> list[tuple[np.ndarray, dict[str, float], dict[str, float]]]:
    shuffled_observations = [obs for obs in observations]
    random.shuffle(shuffled_observations)
    return shuffled_observations


@pytest.fixture
def observations_with_other_images_size(
    observations,
) -> list[tuple[np.ndarray, dict[str, float], dict[str, float]]]:
    other_size = tuple(np.array(observations[0][0].shape) + 1)
    other_size_image = np.full(other_size, 100)
    return [
        (other_size_image, features, actions) for _, features, actions in observations
    ]


def test_recordings_service(my_service: RecorderDataService, observations) -> None:
    game = "trackmania"
    user = "pytest"
    name = "test"
    fps = 10

    my_service.start_streaming(game, user, name, fps)
    for image, features, actions in observations:
        my_service.put_observation(image, features, actions)
    my_service.stop_streaming()

    recordig = my_service.get_recording(game, user, name)

    assert recordig.fps == fps
    assert len(recordig.images) == len(observations)
    assert len(recordig.features) == len(observations)
    assert len(recordig.actions) == len(observations)

    for i, (image, features, actions) in enumerate(observations):
        assert (recordig.images[i] == image).all()
        assert list(recordig.features[i]) == list(features.values())
        assert list(recordig.actions[i]) == list(actions.values())


def test_adding_to_recordings_container(
    my_service: RecorderDataService,
    observations,
    shuffled_observations,
    observations_with_other_images_size,
) -> None:
    game = "trackmania"
    user = "pytest"

    recordings: list[Recording] = []
    for i, (test_observations, fps) in enumerate(
        [
            (observations, 10),
            (shuffled_observations, 10),
            (observations, 100),
            (observations_with_other_images_size, 10),
        ]
    ):
        name = f"test_{i}"
        my_service.start_streaming(game, user, name, fps)
        for image, features, actions in test_observations:
            my_service.put_observation(image, features, actions)
        my_service.stop_streaming()
        recordings.append(my_service.get_recording(game, user, name))

    recordings_container = RecordingsContainer()

    for recording in recordings:
        assert recordings_container.can_be_added(recording)

    assert recordings_container.try_add(recordings[0])
    assert recordings_container.try_add(recordings[1])
    assert not recordings_container.try_add(recordings[2])
    assert not recordings_container.try_add(recordings[3])


def test_iteration_over_recordings_container(
    my_service: RecorderDataService,
    observations,
    shuffled_observations,
    observations_with_other_images_size,
) -> None:
    game = "trackmania"
    user = "pytest"
    fps = 10

    recordings_container = RecordingsContainer()

    for i, test_observations in enumerate(
        [observations, shuffled_observations, observations_with_other_images_size]
    ):
        name = f"test_{i}"
        my_service.start_streaming(game, user, name, fps)
        for image, features, actions in test_observations:
            my_service.put_observation(image, features, actions)
        my_service.stop_streaming()
        recordings_container.try_add(my_service.get_recording(game, user, name))

    expected_items_number = len(observations) + len(shuffled_observations)
    actual_items_number = len([item for item in recordings_container.get_all()])
    assert actual_items_number == expected_items_number
