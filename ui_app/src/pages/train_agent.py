import streamlit as st

from ui_app.src.forms import (
    configure_game,
    configure_env,
    configure_training,
    review_config,
    start_training,
)
from ui_app.src.page_layout import racing_toolbox_page_layout
from ui_app.src.shared import Shared


def main():
    configure_from_scratch()


def configure_from_scratch():
    st.header("Configure training")
    shared = Shared()
    game_config = configure_game()
    env_config = configure_env(game_config)
    training_config = configure_training()
    if game_config and env_config and training_config:
        with st.sidebar.header("Review configuration"):
            game_config = review_config(
                game_config, "Game configuration", shared.games_source
            )
            env_config = review_config(
                env_config, "Environment configuration", shared.envs_source
            )
            training_config = review_config(
                training_config,
                "Training configuration",
                shared.training_configs_source,
            )
        wandb_api_key = shared.user_data.wandb_api_key
        start_training(game_config, env_config, training_config, wandb_api_key)


if __name__ == "__main__":
    racing_toolbox_page_layout("Automobile Agent Training Starter", main)
