import streamlit as st
import json

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig

from ui_app.src.config_source.abstract import (
    AbstractConfigSource,
    RacingToolboxConfiguration,
)
from ui_app.src.shared import Shared
from ui_app.src.utils import SetEncoder


def review_all(
    game_config: GameConfiguration,
    env_config: EnvConfig,
    training_config: TrainingConfig,
) -> tuple[GameConfiguration, EnvConfig, TrainingConfig]:
    with st.sidebar.header("Review configuration"):
        shared = Shared()
        return (
            review_config(game_config, "Game configuration", shared.games_source),
            review_config(env_config, "Environment configuration", shared.envs_source),
            review_config(
                training_config,
                "Training configuration",
                shared.training_configs_source,
            ),
        )


def review_config(
    config: RacingToolboxConfiguration,
    config_label: str,
    config_source: AbstractConfigSource,
) -> RacingToolboxConfiguration:
    st.sidebar.markdown("""---""")
    config_cls = type(config)
    selected = st.sidebar.selectbox(config_label, options=["Review", "Edit", "Upload"])
    if selected == "Review":
        st.sidebar.json(config.dict(), expanded=False)
    elif selected == "Edit":
        config_str = st.text_area(
            f"Edit {config_label}",
            json.dumps(config.dict(), cls=SetEncoder, indent=4),
            height=200,
        )
        config_name = st.sidebar.text_input(
            f"Provide new {config_label} name", value="My_New_config"
        )
        try:
            new_config = config_cls(**json.loads(config_str))
        except:
            st.warning(f"Oops. Your edited {config_label} is not valid")
            return config
        if st.sidebar.button("Submit"):
            config_source.add_config(config_name, new_config)
            config = new_config
    else:
        st.sidebar.write(f"Upload {config_label}")
        uploaded = st.sidebar.file_uploader(
            f"Upload/select {config_label} config JSON data", type=["JSON"]
        )
        if uploaded:
            config_name = st.sidebar.text_input(
                f"Provide new {config_label} name", value="My_New_config"
            )
            try:
                new_config = config_cls(**json.load(uploaded))
            except:
                st.warning(f"Oops. Your uploaded {config_label} is not valid")
                return config
            if st.sidebar.button("Submit"):
                config_source.add_config(config_name, new_config)
                config = new_config
    return config
