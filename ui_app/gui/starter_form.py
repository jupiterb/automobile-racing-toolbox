import streamlit as st

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig
from racing_toolbox.training.config.validation import ValidationError, ConfigValidator


def starter_form(
    game_config: GameConfiguration,
    env_config: EnvConfig,
    training_config: TrainingConfig,
):
    st.markdown("""---""")
    st.header("Confirm start of training")
    # TODO get real list of workers
    workers = ["Worker 1", "Worker 2", "Worker 3"]
    workers = st.multiselect("Select rollout workers", options=workers)
    if not any(workers):
        st.write("You must select workers")
        return
    try:
        ConfigValidator().validate(game_config, env_config, training_config)
    except ValidationError as validation:
        st.write("There are some errors in your configuration, resolve them first.")
        for error in validation.errors:
            st.warning(error)
        return
    if st.button("Run"):
        # TODO run and open all trainings page
        pass
