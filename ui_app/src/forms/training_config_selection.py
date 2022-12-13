import streamlit as st
from typing import Optional

from racing_toolbox.training.config import (
    TrainingConfig,
    DQNConfig,
    SACConfig,
    BCConfig,
    AlgorithmConfig,
    ReplayBufferConfig,
)

from ui_app.src.shared import Shared
from ui_app.src.forms.common import select_recordings, configure_model


def configure_training() -> Optional[TrainingConfig]:
    source = Shared().training_configs_source
    training_configs = source.get_configs()
    selected = st.selectbox(
        "Select training configuration", list(training_configs.keys())
    )
    with st.expander("Add new training configuration"):
        name = st.text_input(
            "Provide new training configuration name",
            value="My_New_Training_Config",
        )
        selected_algo = st.selectbox("Select algorithm", ["DQN", "SAC", "BC"])
        with st.form("new_training_conf"):
            algo = configure_algo(selected_algo)
            model = configure_model()
            config = create_new_training_config(algo, model)
            config.algorithm = algo
            submitted = st.form_submit_button("Submit")
            if submitted:
                source.add_config(name, config)
    if selected is not None:
        config = training_configs[selected]
        return config


def create_new_training_config(algo, model, workers=1) -> TrainingConfig:
    st.markdown("""---""")
    st.write("Configure training")
    config = TrainingConfig(
        num_rollout_workers=workers,
        rollout_fragment_length=256 // workers,
        model=model,
        algorithm=algo,
    )
    config.gamma = st.number_input(
        "Gamma", min_value=1e-3, max_value=1.0, value=0.99, step=0.005, format="%.3f"
    )
    config.lr = st.number_input(
        "Learning rate",
        min_value=1e-5,
        max_value=1.0,
        value=5e-4,
        step=5e-5,
        format="%.5f",
    )
    config.train_batch_size = st.number_input(
        "Train batch size", min_value=10, max_value=1000, value=200, step=10
    )
    config.max_iterations = st.number_input(
        "Iterations limit", min_value=1, max_value=1000, value=10
    )
    return config


def configure_algo(algo_name: str) -> AlgorithmConfig:
    if algo_name is not None:
        st.write(f"Choose parameters for {algo_name}")
    buffer = ReplayBufferConfig(capacity=1_000)
    if algo_name == "DQN":
        config = DQNConfig(replay_buffer_config=buffer)
        config.Config.frozen = False
        config.v_max = st.number_input(
            "Maximal v", min_value=0, max_value=100, value=10
        )
        config.v_min = st.number_input(
            "Minimal v", min_value=-100, max_value=0, value=-10
        )
        config.dueling = st.checkbox("Apply duelling")
        config.double_q = st.checkbox("Apply double Q")
    elif algo_name == "SAC":
        config = SACConfig(replay_buffer_config=buffer)
        config.Config.frozen = False
        config.twin_q = st.checkbox("Apply twin Q")
        config.tau = st.number_input(
            "Tau", min_value=1e-4, max_value=1.0, value=5e-3, step=5e-4, format="%.4f"
        )
        config.initial_alpha = st.number_input(
            "Initial alpha",
            min_value=1e-3,
            max_value=1.0,
            value=1.0,
            step=5e-2,
            format="%.3f",
        )
    else:
        config = BCConfig()
        select_recordings()
    return config
