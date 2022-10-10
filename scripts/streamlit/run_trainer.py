import functools
from http import HTTPStatus
from pydantic import ValidationError
import streamlit as st
import json
import ipaddress
import requests

from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training import Trainer, config
from racing_toolbox.training.config.user_defined import TrainingConfig


def get_files():
    with st.sidebar.header("Source data selection"):
        st.sidebar.write("Game Configuration")
        game_data = st.sidebar.file_uploader(
            "Upload/select game config JSON data", type=["JSON"]
        )
        st.sidebar.write("Training Configuration")
        training_data = st.sidebar.file_uploader(
            "Upload/select training config JSON data", type=["JSON"]
        )
        st.sidebar.write("Enviroment Configuration")
        env_data = st.sidebar.file_uploader(
            "Upload/select enviroment config JSON data", type=["JSON"]
        )

    if game_data:
        try:
            game_config = GameConfiguration(**json.load(game_data))
            if "game_conf" not in st.session_state:
                st.session_state["game_conf"] = game_config
        except ValidationError as e:
            st.error(f"Invalid game configuration. {e}")
    else:
        st.error("Please select game config")

    if env_data:
        try:
            env_config = EnvConfig(**json.load(env_data))
            if "env_config" not in st.session_state:
                st.session_state["env_config"] = env_config
        except ValidationError as e:
            st.error(f"Invalid env configuration. {e}")
    else:
        st.error("Please select env config")

    if training_data:
        try:
            training_config = TrainingConfig(**json.load(training_data))
            if "training_config" not in st.session_state:
                st.session_state["training_config"] = training_config
        except ValidationError as e:
            st.error(f"Invalid training configuration. {e}")
    else:
        st.error("Please select training config")


def syn_with_worker(c, i):
    """
    Disable worker address input from changing, send request to worker,
    if not successfull show error msg and enable input
    """
    if f"worker{i}_synced" not in st.session_state:
        st.session_state[f"worker{i}_synced"] = False

    st.session_state[f"address{i}_disabled"] = True
    st.session_state[f"disable_sync{i}"] = True
    address = st.session_state[f"address{i}"] + "/worker/sync"
    response = requests.get(
        address,
        json={
            "game_config": st.session_state["game_config"].json(),
            "env_config": st.session_state["env_config"].json(),
        },
    )
    if response.status_code == HTTPStatus.OK.value:
        c.write("Synced succesfully")
        st.session_state[f"worker{i}_synced"] = True
    else:
        st.session_state[f"address{i}_disabled"] = False
        st.session_state[f"disable_sync{i}"] = False
        c.write(f"unsuccessfull sync: {response.content}, {response.status_code}")


def setup_workers(cols):
    def get_address(c, i):
        if f"address{i}_disabled" not in st.session_state:
            st.session_state[f"address{i}_disabled"] = False

        address = c.text_input(
            "Worker ip address",
            key=f"ip{i}",
            disabled=st.session_state[f"address{i}_disabled"],
        )
        port = c.number_input(
            "Worker port number",
            key=f"port{i}",
            min_value=8000,
            max_value=8100,
            step=1,
            value=8080,
            disabled=st.session_state[f"address{i}_disabled"],
        )
        if address and port:
            try:
                ip = ipaddress.ip_address(address)
                setattr(st.session_state, f"address{i}", f"{ip}:{port}")
            except ValueError:
                c.error(f"IP address {address} is not valid")

    for i, c in enumerate(cols):
        get_address(c, i)
        if f"address{i}" in st.session_state:
            if f"disable_sync{i}" not in st.session_state:
                st.session_state[f"disable_sync{i}"] = False
            clbck = functools.partial(syn_with_worker, c, i)
            c.button(
                f"sync with {i}",
                on_click=clbck,
                disabled=st.session_state[f"disable_sync{i}"],
            )


def training_panel():
    def run_training():
        st.session_state["training_running"] = True 
        ... # TODO: start training in new process, iterate over each worker and send request with own address and port

    address = st.text_input(
        "trainer host",
        key=f"trainer_host",
        value="0.0.0.0",
        disabled=st.session_state["training_running"],
    )
    port = st.number_input(
        "trainer port",
        key=f"trainer_port",
        min_value=8000,
        max_value=8100,
        step=1,
        value=8000,
        disabled=st.session_state["training_running"],
    )
    if address and port:
        try:
            ip = ipaddress.ip_address(address)
            setattr(st.session_state, f"trainer_address", f"{ip}:{port}")
        except ValueError:
            st.error("IP address {address} is not valid")
    if "trainer_address" in st.session_state:
        st.button("run_training", key="training_running", on_click=run_training)


def main():
    get_files()
    if "training_config" in st.session_state:
        training_config = st.session_state["training_config"]
        cols = st.columns(training_config.num_rollout_workers)
        synced_workers = 0
        for i, c in enumerate(cols):
            c.write(f"Worker {i}")
        if "env_config" in st.session_state:
            setup_workers(cols)
        if st.session_state.get("worker{i}_synced"):
            synced_workers += 1
        if synced_workers == training_config.num_rollout_workers:

        "session:", st.session_state


if __name__ == "__main__":
    st.set_page_config(page_title="Automobile training starter", layout="wide")
    main()
