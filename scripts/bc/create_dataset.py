import argparse
import json

from racing_toolbox.datatool import DatasetContainer
from racing_toolbox.datatool.datasets import FromMemoryDataset
from racing_toolbox.datatool.preproc import make_rllib_dataset
from racing_toolbox.datatool.utils import DatasetBasedEnv

from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.observation.utils.ocr import (
    OcrToolConfiguration,
    OcrTool,
    SevenSegmentsOcr,
)


def main():
    args = get_cli_args()

    container = DatasetContainer()
    for name in args.recordings:
        recording = FromMemoryDataset(args.recordings_root, args.game, args.user, name)
        if not container.try_add(recording):
            print(f"Dataset {name} invalid")

    ocr_config: OcrToolConfiguration
    env_config: EnvConfig

    with (open(args.ocr_config) as op, open(args.env_config) as ep):
        ocr_config = OcrToolConfiguration(**json.load(op))
        env_config = EnvConfig(**json.load(ep))

    ocr_tool = OcrTool(ocr_config, SevenSegmentsOcr)
    dataset_env = DatasetBasedEnv(container, ocr_tool)

    dataset_path, configuration_path = make_rllib_dataset(
        dataset_env,
        env_config,
        args.datasets_root,
        args.game,
        args.user,
        args.dataset_name,
    )

    print(
        f"Dataset stored in {dataset_path}, copy of env configuration in {configuration_path}"
    )


def get_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env_config",
        type=str,
        default="./config/trackmania/env.json",
        help="Path to json with env configuartion",
    )
    parser.add_argument(
        "--ocr_config",
        type=str,
        default="./config/trackmania/ocr_config.json",
        help="Path to json with OCR configuartion",
    )
    parser.add_argument(
        "--recordings_root",
        type=str,
        default="./recordings",
        help="Path to recordings root",
    )
    parser.add_argument(
        "--datasets_root", type=str, default="./datasets", help="Path to datasets root"
    )
    parser.add_argument(
        "--game", type=str, default="trackmania", help="Name of the game"
    )
    parser.add_argument("--user", type=str, help="User name")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset. "
        "data.json and config.json will be stored in datasets_root/game/user/dataset_name/ folder",
    )
    parser.add_argument(
        "--recordings",
        type=str,
        nargs="+",
        help="Name of recordings, without .h5 exstension. "
        "They should be stored in recordings_root/game/user/ folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
