import streamlit as st
from typing import Optional
import itertools

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import (
    EnvConfig,
    ActionConfig,
    ObservationConfig,
    RewardConfig,
)
from racing_toolbox.observation.utils.screen_frame import ScreenFrame
from racing_toolbox.observation.config import LidarConfig, TrackSegmentationConfig

from ui_app.config_source.file_based import FileSysteConfigSource


def actions_form(
    game_config: Optional[GameConfiguration], discrete_action_space: bool
) -> ActionConfig:
    action_config = ActionConfig(available_actions=None)
    if discrete_action_space:
        if game_config is not None:
            combinations = []
            for L in range(len(game_config.discrete_actions_mapping) + 1):
                for subset in itertools.combinations(
                    list(game_config.discrete_actions_mapping.keys()), L
                ):
                    combinations.append(subset)
            actions = st.multiselect(
                "Choose discrete actions combinations", combinations
            )
            action_config.available_actions = {}
            for action in game_config.discrete_actions_mapping:
                action_config.available_actions[action] = set()
                for i, subset in enumerate(actions):
                    if action in set(subset):
                        action_config.available_actions[action].add(i)
        else:
            st.warning(
                "Cannot select discrete actions if game configuration is not selected."
            )
        st.markdown("""---""")
    return action_config


def screen_frame_form() -> ScreenFrame:
    st.write("Configure screen frame")
    return ScreenFrame(
        top=st.number_input("Top", min_value=0.0, max_value=1.0, value=0.0, step=0.05),
        bottom=st.number_input(
            "Bottom", min_value=0.0, max_value=1.0, value=1.0, step=0.05
        ),
        left=st.number_input(
            "Left", min_value=0.0, max_value=1.0, value=0.0, step=0.05
        ),
        right=st.number_input(
            "Right", min_value=0.0, max_value=1.0, value=1.0, step=0.05
        ),
    )


def lidar_form() -> LidarConfig:
    st.write("Configure LIDAR parameters")
    return LidarConfig(
        depth=st.number_input("Depth", min_value=1, max_value=5, value=3),
        lidar_start=(
            st.number_input(
                "Source Coord Y", min_value=0.0, max_value=1.0, value=0.9, step=0.05
            ),
            st.number_input(
                "Source Coord X", min_value=0.0, max_value=1.0, value=0.5, step=0.05
            ),
        ),
        angles_range=(
            -90,
            90,
            st.number_input("Angles between rays", min_value=1, max_value=30, value=10),
        ),
    )


def track_segmentation_form() -> TrackSegmentationConfig:
    st.write("Provide expected color of track")
    track_color = (
        st.number_input("Red", min_value=0, max_value=256, value=200),
        st.number_input("Green", min_value=0, max_value=256, value=200),
        st.number_input("Blue", min_value=0, max_value=256, value=200),
    )
    return TrackSegmentationConfig(
        track_color=track_color,
        tolerance=st.number_input(
            "Track color tolerance", min_value=10, max_value=200, value=90, step=10
        ),
        noise_reduction=st.number_input(
            "Nose reduction level", min_value=3, max_value=9, value=5, step=2
        ),
    )


def autoencoders_form() -> ScreenFrame:
    # It's now placeholder only and returns ScreenFrame
    vaes = {"VAE 1": ScreenFrame(), "VAE 2": ScreenFrame(top=0.5)}
    if len(vaes) == 0:
        st.write("There are currently no autoencoders. Train one!")
        # TODO form for VAE train
    selected = st.selectbox("Select autencoder", options=list(vaes.keys()))
    return vaes[selected]


def observation_form(feature_extraction_type: str) -> ObservationConfig:
    st.write("Observation config")
    stack_size = st.number_input("Stack size", min_value=1, max_value=10, value=4)
    shape = (84, 84)
    lidar_config = None
    segmentation_config = None
    st.write(f"Configure {feature_extraction_type}")
    if feature_extraction_type == "Reshape":
        shape = (
            st.number_input("Height", min_value=32, max_value=256, value=84),
            st.number_input("Width", min_value=32, max_value=256, value=84),
        )
        frame = screen_frame_form()
    elif feature_extraction_type == "LIDAR":
        lidar_config = lidar_form()
        segmentation_config = track_segmentation_form()
        frame = screen_frame_form()
    else:
        frame = autoencoders_form()
    observation_config = ObservationConfig(
        frame=frame,
        shape=shape,
        stack_size=stack_size,
        lidar_config=lidar_config,
        track_segmentation_config=segmentation_config,
    )
    return observation_config


def reward_form() -> RewardConfig:
    st.write("Reward config")
    speed_diff_thresh = st.number_input(
        "Sppeed difference threshold", min_value=0, max_value=40, value=10
    )
    memory_length = st.number_input(
        "Number of past stpes in difference calculstions",
        min_value=1,
        max_value=20,
        value=8,
    )
    baseline = st.number_input("Baseline", min_value=0, max_value=200, value=100)
    scale = st.number_input("Scale", min_value=1, max_value=1_000, value=300)
    clip_range = (-scale, scale)
    st.write("Where reward = (reward - baseline) / scale")
    return RewardConfig(
        speed_diff_thresh=speed_diff_thresh,
        memory_length=memory_length,
        baseline=baseline,
        scale=scale,
        clip_range=clip_range,
    )


def new_env_form(
    game_config: Optional[GameConfiguration],
    discrete_action_space: bool,
    feature_extraction_type: str,
) -> EnvConfig:
    action_config = actions_form(game_config, discrete_action_space)
    observation_config = observation_form(feature_extraction_type)
    st.markdown("""---""")
    reward_config = reward_form()
    st.markdown("""---""")
    max_episode_length = st.number_input(
        "Maximal episode length", min_value=100, max_value=10_000, value=1_000
    )
    return EnvConfig(
        action_config=action_config,
        observation_config=observation_config,
        reward_config=reward_config,
        max_episode_length=max_episode_length,
    )


def configure_env(game_config: Optional[GameConfiguration]) -> Optional[EnvConfig]:
    source = FileSysteConfigSource[EnvConfig]("./config/envs")
    envs = source.get_configs()
    selected = st.selectbox("Select environment", list(envs.keys()))
    with st.expander("Add new environment"):
        name = st.text_input("Provide new environment name", value="My_New_Env")
        action_space_type = st.selectbox(
            "Select action spave type", ["Continous", "Discrete"]
        )
        discrete_action_space = action_space_type == "Discrete"
        feature_extraction_type = st.selectbox(
            "Select feature extraction type",
            options=["Reshape", "LIDAR", "Autoencoder"],
        )
        with st.form("new_env"):
            config = new_env_form(
                game_config, discrete_action_space, feature_extraction_type
            )
            submitted = st.form_submit_button("Submit")
            if submitted:
                source.add_config(name, config)
    if selected is not None:
        config = envs[selected]
        return config
