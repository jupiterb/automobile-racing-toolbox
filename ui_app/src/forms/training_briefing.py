import streamlit as st

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig
from racing_toolbox.training.config.validation import ValidationError, ConfigValidator

from ui_app.src.shared import Shared


def start_training(
    game_config: GameConfiguration,
    env_config: EnvConfig,
    training_config: TrainingConfig,
    wandb_key: str,
):
    st.markdown("""---""")
    st.header("Confirm start of training")
    try:
        # ConfigValidator().validate(game_config, env_config, training_config)
        pass
    except ValidationError as validation:
        st.write("There are some errors in your configuration, resolve them first.")
        for error in validation.errors:
            st.warning(error)
        return
    if st.button("Run"):
        Shared().trainer_service.start_training(
            game_config, env_config, training_config, wandb_key
        )
        pass


def resume_training(
    training_config: TrainingConfig,
    wandb_key: str,
    wandb_run_reference: str,
    checkpoint_name: str,
):
    st.markdown("""---""")
    st.header("Confirm start of training")
    if st.button("Run"):
        Shared().trainer_service.resume_training(
            training_config, wandb_key, wandb_run_reference, checkpoint_name
        )
        pass


def train_autoencoder():
    st.markdown("""---""")
    st.header("Confirm start of training")
    if st.button("Run"):
        # TODO add endpoint for autoencoder
        pass
