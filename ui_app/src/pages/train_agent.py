import streamlit as st

from ui_app.src.forms import (
    configure_game,
    configure_env,
    configure_training,
    review_config,
)

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig
from racing_toolbox.training.config.validation import ConfigValidator

from ui_app.src.page_layout import racing_toolbox_page_layout
from ui_app.src.shared import Shared


def main():
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


def start_training(
    game_config: GameConfiguration,
    env_config: EnvConfig,
    training_config: TrainingConfig,
    wandb_key: str,
):
    st.markdown("""---""")
    st.header("Confirm start of training")

    validator = ConfigValidator()
    validator.validate_discrete_actions_compatibilty(game_config, env_config)
    validator.validate_continous_actions_compatibilty(game_config, env_config)
    if any(validator.errors):
        st.write("There are some errors in your configuration, resolve them first.")
        for error in validator.errors:
            st.warning(error)
        return

    if st.button("Run"):
        Shared().trainer_service.start_training(
            game_config, env_config, training_config, wandb_key
        )


if __name__ == "__main__":
    racing_toolbox_page_layout("Automobile Agent Training Starter", main)
