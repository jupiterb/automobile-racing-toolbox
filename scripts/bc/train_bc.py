import sys

from racing_toolbox.conf import get_game_config

from racing_toolbox.environment.builder import setup_env
from racing_toolbox.environment.config import (
    EnvConfig,
    ActionConfig,
    RewardConfig,
    ObservationConfig,
)

from racing_toolbox.datatool import DatasetContainer
from racing_toolbox.datatool.datasets import FromMemoryDataset
from racing_toolbox.datatool.preproc import make_rllib_dataset
from racing_toolbox.datatool.utils import DatasetBasedEnv

from racing_toolbox.observation.utils import ScreenFrame
from racing_toolbox.observation.utils.ocr import OcrTool, SevenSegmentsOcr

from racing_toolbox.training import Trainer
from racing_toolbox.training.config import TrainingParams
from racing_toolbox.training.config import ModelConfig
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.training.config import (
    TrainingConfig,
    BCConfig,
    EvalConfig,
)


def main(path_to_data: str, game: str, user: str, data: list[str]):
    game_config = get_game_config()
    env_config = get_env_config()

    bc_dataset_root = "./datasets"
    bc_dataset_name = "bc1"

    if len(data):
        container = DatasetContainer()
        for name in data:
            dataset = FromMemoryDataset(path_to_data, game, user, name)
            if not container.try_add(dataset):
                print(f"Dataset {name} invalid")

        ocr_tool = OcrTool(game_config.ocrs, SevenSegmentsOcr)
        dataset_env = DatasetBasedEnv(container, ocr_tool)

        make_rllib_dataset(
            dataset_env, env_config, bc_dataset_root, game, user, bc_dataset_name
        )

    real_env = setup_env(game_config, env_config)
    train_config = get_train_config(real_env)

    trainer = Trainer(train_config)
    trainer.run()


def get_env_config() -> EnvConfig:
    action_config = ActionConfig(
        available_actions={
            "FORWARD": {0, 1, 2},
            "BREAK": set(),
            "RIGHT": {1, 3},
            "LEFT": {2, 4},
        }
    )

    reward_conf = RewardConfig(
        speed_diff_thresh=3,
        memory_length=2,
        off_track_termination=True,
        clip_range=(-300, 300),
        baseline=20,
        scale=300,
    )

    observation_conf = ObservationConfig(
        frame=ScreenFrame(top=0.475, bottom=0.9125, left=0.01, right=0.99),
        shape=(60, 60),
        stack_size=4,
        lidar_config=None,
        track_segmentation_config=None,
    )

    return EnvConfig(
        action_config=action_config,
        reward_config=reward_conf,
        observation_config=observation_conf,
        max_episode_length=1_000,
    )


def get_train_config(env) -> TrainingParams:
    eval_conf = EvalConfig(
        eval_name="test_evaluation",
        eval_interval_frequency=1,
        eval_duration_unit="timesteps",
        eval_duration=10,
    )

    training_config: TrainingConfig = TrainingConfig(
        num_rollout_workers=0,
        rollout_fragment_length=10,
        train_batch_size=12,
        max_iterations=2,
        algorithm=BCConfig(),
        # offline_data=[Path("./datasets/trackmania/ptr/bc1/data.json")],
        evaluation_config=eval_conf,
        model=ModelConfig(
            fcnet_hiddens=[50],
            fcnet_activation="relu",
            conv_filters=[
                (32, (8, 8), 4),
                (64, (4, 4), 2),
                (64, (3, 3), 1),
                (64, (8, 8), 1),
            ],
        ),
    )

    params = TrainingParams(
        **training_config.dict(),
        input_=["./datasets/trackmania/ptr/bc1/data.json"],
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    return params


if __name__ == "__main__":
    path_to_data = sys.argv[1]
    game = sys.argv[2]
    user = sys.argv[3]
    data = sys.argv[4:]
    main(path_to_data, game, user, data)
