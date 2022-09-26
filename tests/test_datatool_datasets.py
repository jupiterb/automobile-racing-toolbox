import pytest
import random
import shutil
import numpy as np

from racing_toolbox.datatool.datasets import DatasetContainer
from racing_toolbox.datatool.services import InMemoryDatasetService


@pytest.fixture
def path(autouse=True):
    path = "./temp"
    yield path
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def observations() -> list[tuple[np.ndarray, dict[str, float]]]:
    obs_number = lambda i: i % 5 + i % 3 + 1
    image = lambda i: np.full((50, 50, 3), obs_number(i))
    actions = lambda i: {"x": obs_number(i) / 2, "y": -obs_number(i) / 4}
    return [(image(i), actions(i)) for i in range(100)]


@pytest.fixture
def shuffled_observations(
    observations,
) -> list[tuple[np.ndarray, dict[str, float]]]:
    shuffled_observations = [obs for obs in observations]
    random.shuffle(shuffled_observations)
    return shuffled_observations


@pytest.fixture
def observations_with_other_images_size(
    observations,
) -> list[tuple[np.ndarray, dict[str, float]]]:
    other_size = tuple(np.array(observations[0][0].shape) + 1)
    other_size_image = np.full(other_size, 100)
    return [(other_size_image, actions) for _, actions in observations]


def test_dataset_service(path, observations) -> None:
    game = "trackmania"
    user = "pytest"
    name = "test"
    fps = 10

    with InMemoryDatasetService(path, game, user, name, fps) as service:
        for image, actions in observations:
            service.put(image, actions)

    with InMemoryDatasetService.get_dataset(path, game, user, name).get() as dataset:
        assert dataset.fps == fps
        assert len(dataset.observations) == len(observations)
        assert len(dataset.actions) == len(observations)

        for i, (image, actions) in enumerate(observations):
            assert (dataset.observations[i] == image).all()
            assert list(dataset.actions[i]) == list(actions.values())


def test_adding_to_dataset_container(
    path,
    observations,
    shuffled_observations,
    observations_with_other_images_size,
) -> None:
    game = "trackmania"
    user = "pytest"

    empty_dataset_container = DatasetContainer()
    increasing_dataset_container = DatasetContainer()

    for i, (test_observations, fps, should_be_added) in enumerate(
        [
            (observations, 10, True),
            (shuffled_observations, 10, True),
            (observations, 100, False),
            (observations_with_other_images_size, 10, False),
        ]
    ):
        name = f"test_{i}"
        with InMemoryDatasetService(path, game, user, name, fps) as service:
            for image, actions in test_observations:
                service.put(image, actions)
        dataset = InMemoryDatasetService.get_dataset(path, game, user, name)
        assert empty_dataset_container.can_be_added(dataset)
        assert increasing_dataset_container.try_add(dataset) == should_be_added


def test_iteration_over_dataset_container(
    path,
    observations,
    shuffled_observations,
    observations_with_other_images_size,
) -> None:
    game = "trackmania"
    user = "pytest"
    fps = 10

    dataset_container = DatasetContainer()

    for i, test_observations in enumerate(
        [observations, shuffled_observations, observations_with_other_images_size]
    ):
        name = f"test_{i}"
        with InMemoryDatasetService(path, game, user, name, fps) as service:
            for image, actions in test_observations:
                service.put(image, actions)
        dataset = InMemoryDatasetService.get_dataset(path, game, user, name)
        dataset_container.try_add(dataset)

    expected_items_number = len(observations) + len(shuffled_observations)
    actual_items_number = len([item for item in dataset_container.get_all()])
    assert actual_items_number == expected_items_number
