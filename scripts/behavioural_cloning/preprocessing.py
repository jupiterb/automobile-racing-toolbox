from collections import defaultdict, namedtuple
from concurrent.futures import process
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os, logging, cv2, tables


CLIP_INDICES = (380, 730, 10, -10)
N_STACKED = 4
ACTION_OVERHEAD = 1
ACTIONS_ORDER = ["left", "right", "up", "down"]
ACTIONS_FLIPPED = {"left": "right", "right": "left"}
SCALED_MAX_SIDE = 150
CSV_FNAME = "data.csv"
ORIGINAL_SHAPE = (800, 1000)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

def process_dir(dir_path: Path, hdf5_path: Path):

    _, directories, _ = next(os.walk(dir_path))

    target_shape = _get_target_shape(
        ORIGINAL_SHAPE, CLIP_INDICES, SCALED_MAX_SIDE, N_STACKED
    )

    logger.info(f"going to process {len(directories)} episodes")
    logger.info(f"target data shape = {target_shape}, {(0, *target_shape)}")

    hdf5_file = tables.open_file(hdf5_path, mode="w")
    filters = tables.Filters(complevel=5, complib="blosc")
    images_storage = hdf5_file.create_earray(
        hdf5_file.root,
        "images",
        tables.Atom.from_dtype(np.dtype("uint8")),
        shape=(0, *target_shape),
        filters=filters,
        expectedrows=7_000 * len(directories),  # just a guess
    )
    actions_storage = hdf5_file.create_earray(
        hdf5_file.root,
        "actions",
        tables.Atom.from_dtype(np.dtype("uint8")),
        shape=(0, len(ACTIONS_ORDER)),
        filters=filters,
        expectedrows=7_000 * len(directories),  # just a guess
    )

    for episode_dir in tqdm(directories):
        episode_path = dir_path / episode_dir

        logger.info(f"Processing episode from {episode_dir}")

        try:
            X, y = get_episode(episode_path, episode_path / CSV_FNAME)
        except Exception as e:
            logger.error(e)
            continue

        logger.info(f"read episode of shape {X.shape}, and actions of length {len(y)}")

        X, y = process_episode(
            X,
            y,
            CLIP_INDICES,
            ACTIONS_ORDER,
            ACTIONS_FLIPPED,
            N_STACKED,
            ACTION_OVERHEAD,
            SCALED_MAX_SIDE,
        )

        logger.info(f"processed episode, final shape = {X.shape}, actions length = {len(y)}")

        images_storage.append(X)
        actions_storage.append(y)
        del X
        del y
        logger.info("episode saved")

    hdf5_file.close()


def get_episode(episode_dir: Path, csv_path: Path) -> tuple[np.ndarray, list[frozenset]]:
    data_frame = pd.read_csv(csv_path)
    X = []
    y = []

    for _, row in tqdm(data_frame.iterrows()):
        screenshot_path = episode_dir / row["screenshot_file"]
        if not os.path.exists(screenshot_path):
            logger.warning(f"Skipped {screenshot_path} sa it was not present")
            continue

        with Image.open(screenshot_path) as screenshot:
            keys: set[str] = {key for key in row["keys"].split("+") if len(key) > 0}
            y.append(frozenset(keys))
            X.append(np.array(screenshot))

    return np.array(X, dtype=np.uint8), y


def process_episode(
    X: np.ndarray,
    y: list[frozenset],
    clip_indices: tuple,
    one_hot_action_mapping: list,
    action_flipped: dict,
    n_stacked: int,
    action_overhead: int,
    max_side_len: int,
):
    a, b, c, d = clip_indices
    X = _to_grayscale(X[:, 380:730, 10:-10])
    X_rescaled = []
    for i in range(X.shape[0]):
        X_rescaled.append(_rescale(X[i], max_side_len))
    X = np.array(X_rescaled, dtype=np.uint8)

    # add action overhead
    y = y[action_overhead:]
    X = X[:-action_overhead]

    logger.info(f"after action overhead: y.shape = {len(y)}, X.shape = {X.shape}")

    # stack frame with previos n_stacked frames
    frames_stacked = []

    for i in range(n_stacked, X.shape[0]):
        stacked = np.stack(X[i - n_stacked : i], axis=-1)
        frames_stacked.append(stacked)

    X =  np.array(frames_stacked, dtype=np.uint8)
    y = y[n_stacked:]
    logger.info(f"frames stucked, X.len = {len(frames_stacked)}, y.len = {len(y)}")

    X, y = _flip_vertically(X, y, action_flipped)

    y = _one_hot(y, one_hot_action_mapping)

    return X, y


def _to_grayscale(frame):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_image = np.dot(frame, rgb_weights)
    return grayscale_image


def _rescale(frame, max_side_len):
    scale = max_side_len / np.max(frame.shape)
    target_shape = list((scale * np.array(frame.shape)).astype(np.uint8))
    return cv2.resize(frame, dsize=target_shape[::-1], interpolation=cv2.INTER_CUBIC)


def _flip_vertically(X, y, action_flipped: dict):
    def flip_actions(actions: frozenset):
        return frozenset(map(lambda x: action_flipped.get(x, x), actions))

    y_flipped = list(map(flip_actions, y))
    X_flipped = np.flip(X, axis=2)
    return np.vstack((X, X_flipped)), y + y_flipped


def _one_hot(y: list[frozenset], keys_ordered: list):
    return [[int(k in keys) for k in keys_ordered] for keys in y]


def _get_target_shape(original_shape, clip_indices, scaled_max_side, n_stacked):
    a, b, c, d = clip_indices
    clipped_shape = np.empty(shape=original_shape)[a:b, c:d].shape
    scale = scaled_max_side / np.max(clipped_shape)
    target_shape = list((scale * np.array(clipped_shape)).astype(np.uint8))
    return *target_shape, n_stacked


if __name__ == "__main__":
    process_dir(
        Path("C:\\Users\czyjt\Projects\engineering-thesis\\automobile-racing-toolbox\\appserver\episode\dataservice\data\\trackmania"),
        Path("compressed_dataset.hdf5")
    )
