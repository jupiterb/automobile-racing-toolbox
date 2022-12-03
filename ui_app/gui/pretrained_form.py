import streamlit as st
from typing import Optional

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import TrainingConfig


from ui_app.gui.training_form import algo_form, training_form
from ui_app.trainings_source.file_based import FileSysteModelsSource, WeightsAndConfigs


def select_pretained_model() -> Optional[WeightsAndConfigs]:
    source = FileSysteModelsSource("./config/pretrained")
    models = source.get_models()
    if not len(models):
        st.write(
            "There are currently no pre-trained models. Start training from scratch and create one!"
        )
        return None
    selected = st.selectbox("Select pretarined model", list(models.keys()))
    if selected is None:
        return None
    return models[selected]


def configure_pretained() -> Optional[
    tuple[GameConfiguration, EnvConfig, TrainingConfig]
]:
    selected = select_pretained_model()
    if selected is None:
        return None
    weights, model_config, game_config, env_config = selected
    st.markdown("""---""")
    selected_algo = st.selectbox("Select algorithm", ["DQN", "SAC", "BC"])
    algo = algo_form(selected_algo)
    training_config = training_form(algo, model_config)
    training_config.model = model_config
    # TODO apply weights
    return game_config, env_config, training_config
