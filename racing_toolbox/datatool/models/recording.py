from numpy import ndarray


class Recording:
    game: str
    user: str
    name: str
    fps: int
    images: ndarray
    actions: ndarray
    features: ndarray
    actions_names: set[str]
    features_names: set[str]
