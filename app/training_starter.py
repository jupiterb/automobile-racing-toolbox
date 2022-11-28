import streamlit as st
import streamlit_pydantic as sp
import typing
from typing import Optional
import itertools

from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.environment.config import EnvConfig
from racing_toolbox.training.config import (
    TrainingConfig,
    ModelConfig,
    DQNConfig,
    SACConfig,
    BCConfig,
    AlgorithmConfig,
    ReplayBufferConfig,
)
from racing_toolbox.training.config.user_defined import Activation

from app.config_source.game.file_system_based import FileSystemGameConfigSource


def select_training_way() -> bool:
    """returns True if user wants to run training from scratch, else False"""
    from_sratch = True
    st.header("Select way to train new amazing AI")
    option = st.selectbox(
        "Would you like to use pretrained model or create new one from scratch?",
        ["From scratch", "Use pretrained"],
    )
    from_sratch = option == "From scratch"
    return from_sratch


def configure_algo() -> AlgorithmConfig:
    seleced_algo = st.selectbox("Select algorithm", ["DQN", "SAC", "BC"])
    st.markdown("""---""")
    if seleced_algo is not None:
        st.write(f"Choose parameters for {seleced_algo}")
    buffer = ReplayBufferConfig(capacity=1_000)
    if seleced_algo == "DQN":
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
    elif seleced_algo == "SAC":
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
    return config


def configure_model() -> ModelConfig:
    st.markdown("""---""")
    st.write("Configure model architecture")
    model = ModelConfig(
        fcnet_activation="relu",
        conv_activation="relu",
        fcnet_hiddens=[100],
        conv_filters=[],
    )
    avtivations = typing.get_args(Activation)
    conv_activation_option = st.selectbox(
        "Select activation function for conv", avtivations
    )
    if conv_activation_option is not None:
        model.conv_activation = conv_activation_option
    # TODO adding conv layers
    fcnet_activation_option = st.selectbox(
        "Select activation function for fcnet", avtivations
    )
    if fcnet_activation_option is not None:
        model.fcnet_activation = fcnet_activation_option
    # TODO adding fcnet layers
    return model


def configure_training(algo, model, workers=1) -> TrainingConfig:
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


def configure_game() -> Optional[GameConfiguration]:
    source = FileSystemGameConfigSource("./config/games")
    games = source.get_configs()
    selected = st.selectbox("Select game", list(games.keys()))
    if st.button("Or add new one"):
        config = sp.pydantic_form(
            key="new game config",
            model=GameConfiguration,
            submit_label="Add",
            clear_on_submit=True,
        )
        if config is not None:
            source.add_config(config)
    if selected is not None:
        config = games[selected]
        return config


def configure_env(game_config: Optional[GameConfiguration]) -> Optional[EnvConfig]:
    st.markdown("""---""")
    st.write("Action config")
    if st.checkbox("Use discrete action space") and game_config is not None:
        combinations = []
        for L in range(len(game_config.discrete_actions_mapping) + 1):
            for subset in itertools.combinations(
                list(game_config.discrete_actions_mapping.keys()), L
            ):
                combinations.append(subset)
        actions = st.multiselect("Choose discrete actions combinations", combinations)


def configure_from_scratch() -> bool:
    st.markdown("""---""")
    st.header("Configure your environment")
    game_config = configure_game()
    env_config = configure_env(game_config)

    st.header("Configure training")
    algo = configure_algo()
    model = configure_model()
    configure_training(algo, model)
    # TODO build training params

    # TODO model arch
    # TODO return True if configured
    return True


def configure_with_pretrained() -> bool:
    st.markdown("""---""")
    st.header("Select pretrained model")
    # TODO pretreined model source (with env and game)

    st.header("Configure training")
    algo = configure_algo()
    # TODO get model from pretrained

    # TODO return True if configured
    return True


def main():
    st.set_page_config(page_title="Automobile Training Starter", layout="wide")
    st.markdown(
        f'<h1 style="color:#ff8833;font-size:40px;">{"Automobile Training Starter"}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("""---""")

    from_scratch = select_training_way()
    if from_scratch:
        configure_from_scratch()
    else:
        configure_with_pretrained()


if __name__ == "__main__":
    main()
