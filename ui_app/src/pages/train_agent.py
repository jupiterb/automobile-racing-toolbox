import streamlit as st

from ui_app.src.forms import (
    configure_game,
    configure_env,
    configure_training,
    review_all,
    start_training,
)
from ui_app.src.page_layout import racing_toolbox_page_layout


def main():
    configure_from_scratch()


def configure_from_scratch():
    st.header("Configure training")
    game_config = configure_game()
    env_config = configure_env(game_config)
    training_config = configure_training()
    if game_config and env_config and training_config:
        game_config, env_config, training_config = review_all(
            game_config, env_config, training_config
        )
        # TODO get user api key
        start_training(game_config, env_config, training_config, "USER KEY")


if __name__ == "__main__":
    racing_toolbox_page_layout("Automobile Agent Training Starter", main)
