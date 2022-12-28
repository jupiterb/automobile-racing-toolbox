import streamlit as st

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig
from racing_toolbox.training.config.validation import ValidationError, ConfigValidator
from racing_toolbox.observation.config.vae_config import (
    VAETrainingConfig,
    VAEModelConfig,
)

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


def train_autoencoder(
    wandb_key: str,
    training_params: VAETrainingConfig,
    encoder_config: VAEModelConfig,
    bucket_name: str,
    recordings_refs: list[str],
):
    st.markdown("""---""")
    st.header("Confirm start of training")
    if st.button("Run"):
        Shared().trainer_service.start_autoencoder_training(
            wandb_key, training_params, encoder_config, bucket_name, recordings_refs
        )
