import functools
from http import HTTPStatus
from multiprocessing import Process
from pydantic import ValidationError
import streamlit as st
import json
import ipaddress
import requests

from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training import Trainer, config, trainer
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.training.config.user_defined import TrainingConfig
from racing_toolbox.environment import builder
from ray.rllib.env.policy_server_input import PolicyServerInput


def config_panel() -> bool:
    """loads game, training, env configs to session state. Returns True if its ready, False otherwise"""
    # TODO: make configs disappear from session_state after file was removed
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
            st.session_state["game_config"] = game_config
        except ValidationError as e:
            st.error(f"Invalid game configuration. {e}")
    else:
        st.error("Please select game config")

    if env_data:
        try:
            env_config = EnvConfig(**json.load(env_data))
            st.session_state["env_config"] = env_config
        except ValidationError as e:
            st.error(f"Invalid env configuration. {e}")
    else:
        st.error("Please select env config")

    if training_data:
        try:
            training_config = TrainingConfig(**json.load(training_data))
            st.session_state["training_config"] = training_config
        except ValidationError as e:
            st.error(f"Invalid training configuration. {e}")
    else:
        st.error("Please select training config")

    return all(
        c in st.session_state for c in ["game_config", "training_config", "env_config"]
    )


def trainer_panel() -> bool:
    """Sets up trainer address. Returns True if all data was submitted, False otherwise"""
    st.write("Trainer")
    address = st.text_input(
        "trainer host",
        key=f"trainer_host",
        value="0.0.0.0",
    )
    port = st.number_input(
        "trainer port",
        key=f"trainer_port",
        min_value=8000,
        max_value=8100,
        step=1,
        value=8000,
    )
    if address and port:
        try:
            ip = ipaddress.ip_address(address)
            st.session_state[f"trainer_address"] = f"{ip}:{port}"
        except ValueError:
            st.error("IP address {address} is not valid")

    return "trainer_address" in st.session_state


def workers_panel() -> bool:
    """fetch workers data from user, and sync configuration between trainer and workers.
    Returns True if all workers were synced, False otherwise"""

    def get_address(c, i):
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
            value=8000,
            disabled=st.session_state[f"address{i}_disabled"],
        )
        if address and port:
            try:
                ip = ipaddress.ip_address(address)
                return f"{ip}:{port}"
            except ValueError:
                c.error(f"IP address {address} is not valid")

    def syn_with_worker(c, i):
        """
        Disable worker address input from changing, send request to worker,
        if not successfull show error msg and enable input
        """
        address = "http://" + st.session_state[f"address{i}"] + "/worker/sync"
        try:

            response = requests.post(
                address,
                json={
                    "game_config": json.loads(st.session_state["game_config"].json()),
                    "env_config": json.loads(st.session_state["env_config"].json()),
                    "policy_address": st.session_state["trainer_address"].split(":"),
                },
            )
            if response.status_code == HTTPStatus.OK.value:
                c.write("Synced succesfully")
                return True
            else:
                c.write(
                    f"unsuccessfull sync: {response.content}, {response.status_code}"
                )
                return False
        except requests.ConnectionError as exc:
            c.write(f"unable to establish connection with worker")
            return False

    # setup shared state variables
    for i in range(st.session_state["training_config"].num_rollout_workers):
        if f"address{i}_disabled" not in st.session_state:
            st.session_state[f"address{i}_disabled"] = False
        if f"worker{i}_synced" not in st.session_state:
            st.session_state[f"worker{i}_synced"] = False
        if f"disable_sync{i}" not in st.session_state:
            st.session_state[f"disable_sync{i}"] = True

    cols = st.columns(st.session_state["training_config"].num_rollout_workers)
    for i, c in enumerate(cols):
        c.write(f"Worker {i}")
        if address := get_address(c, i):
            st.session_state[f"address{i}"] = address
            st.session_state[f"disable_sync{i}"] = False
        if (
            c.button(
                f"sync with {i}",
                disabled=st.session_state[f"disable_sync{i}"],
            )
            and address
        ):
            st.session_state[f"address{i}_disabled"] = True
            st.session_state[f"disable_sync{i}"] = True
            if syn_with_worker(c, i):
                st.session_state[f"worker{i}_synced"] = True
            else:
                st.session_state[f"address{i}_disabled"] = False
                st.session_state[f"disable_sync{i}"] = False

    return all(
        st.session_state[f"worker{i}_synced"]
        for i in range(st.session_state["training_config"].num_rollout_workers)
    )


def _training_thread(game_config, env_config, training_config, host, port):
    def input_(ioctx):
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                host,
                port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None

    trainer_params = TrainingParams(
        **training_config.dict(),
        env=builder.setup_env(game_config, env_config),
        input_=input_,
    )
    trainer = Trainer(trainer_params)
    trainer.run()


def training_panel():
    def run_training():
        st.session_state["training_running"] = True
        trainer_process = Process(
            target=_training_thread,
            args=(
                st.session_state["game_config"],
                st.session_state["env_config"],
                st.session_state["training_config"],
                st.session_state["trainer_host"],
                st.session_state["trainer_port"],
            ),
        )
        st.session_state["trainer_process"] = trainer_process
        trainer_process.start()

    def stop_training():
        st.session_state["trainer_process"].kill()
        st.session_state["trainer_process"].join()
        st.session_state["training_running"] = False

    # initialize shared state attributes
    if "training_running" not in st.session_state:
        st.session_state["training_running"] = False
    if "trainer_process" not in st.session_state:
        st.session_state["trainer_process"] = None

    st.button(
        "run_training",
        disabled=st.session_state["training_running"],
        on_click=run_training,
    )
    st.button(
        "stop training",
        disabled=not st.session_state["training_running"],
        on_click=stop_training,
    )


def main():
    configs_ready = config_panel()
    trainer_ready = trainer_panel() if configs_ready else False
    workers_ready = workers_panel() if trainer_ready else False
    training_ready = training_panel() if workers_ready and trainer_ready else False
    "session:", st.session_state


if __name__ == "__main__":
    st.set_page_config(page_title="Automobile training starter", layout="wide")
    main()
