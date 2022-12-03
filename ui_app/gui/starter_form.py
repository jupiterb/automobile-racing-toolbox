import streamlit as st


from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig


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
    if any(workers):
        if st.button("Run"):
            # TODO run and open all trainings page
            pass
    else:
        st.write("You must select workers")
