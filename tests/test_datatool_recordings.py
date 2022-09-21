import pytest
import shutil
import numpy as np

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
    return BinaryFileRecordingsService("./temp")


@pytest.fixture
def cases() -> list[tuple[np.ndarray, dict[str, float], dict[str, float]]]:
    test_case_number = lambda i: i % 5 + i % 3 + 1
    image = lambda i: np.full((50, 50, 3), test_case_number(i))
    features = lambda i: {"a": test_case_number(i), "b": -test_case_number(i)}
    actions = lambda i: {"x": test_case_number(i) / 2, "y": -test_case_number(i) / 4}
    return [(image(i), features(i), actions(i)) for i in range(100)]


def test_recordings_service(my_service: RecorderDataService, cases) -> None:
    game = "trackmania"
    user = "pytest"
    name = "test"
    fps = 10

    my_service.start_streaming(game, user, name, fps)
    for image, features, actions in cases:
        my_service.put_observation(image, features, actions)
    my_service.stop_streaming()

    recordig = my_service.get_recording(game, user, name)

    assert recordig.fps == fps
    assert len(recordig.images) == len(cases)
    assert len(recordig.features) == len(cases)
    assert len(recordig.actions) == len(cases)

    for i, (image, features, actions) in enumerate(cases):
        assert (recordig.images[i] == image).all()
        assert list(recordig.features[i]) == list(features.values())
        assert list(recordig.actions[i]) == list(actions.values())
