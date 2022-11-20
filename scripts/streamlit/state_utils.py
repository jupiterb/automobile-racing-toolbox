from __future__ import annotations
from calendar import c
from dataclasses import dataclass, field
from email import policy
import multiprocessing
import requests
import json
from typing import ClassVar, Optional, Generator
from http import HTTPStatus
import streamlit as st

from racing_toolbox.environment.config.env import EnvConfig
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.training.config.user_defined import TrainingConfig
from racing_toolbox.training.config.params import TrainingParams
from racing_toolbox.environment import builder
from ray.rllib.env.policy_server_input import PolicyServerInput
from racing_toolbox.training import Trainer, config, trainer


class SharedState:
    """Proxy for session streamlit.session_state"""

    @property
    def configs_ready(self) -> bool:
        return all(
            c in st.session_state
            for c in ["game_config", "training_config", "env_config"]
        )

    @property
    def trainer_ready(self) -> bool:
        return "trainer_address" in st.session_state

    @property
    def workerset(self) -> Workerset:
        if "workerset" not in st.session_state:
            st.session_state["workerset"] = Workerset()
        return st.session_state["workerset"]

    @property
    def env_config(self) -> EnvConfig:
        return st.session_state["env_config"]

    @env_config.setter
    def env_config(self, v: EnvConfig):
        assert isinstance(v, EnvConfig)
        st.session_state["env_config"] = v

    @property
    def game_config(self) -> GameConfiguration:
        return st.session_state["game_config"]

    @game_config.setter
    def game_config(self, v: GameConfiguration):
        assert isinstance(v, GameConfiguration)
        st.session_state["game_config"] = v

    @property
    def training_config(self) -> TrainingConfig:
        return st.session_state["training_config"]

    @training_config.setter
    def training_config(self, v: TrainingConfig):
        assert isinstance(v, TrainingConfig)
        st.session_state["training_config"] = v

    @property
    def trainer_address(self) -> str:
        return st.session_state["trainer_address"]

    @trainer_address.setter
    def trainer_address(self, v: str):
        assert ":" in v
        st.session_state[f"trainer_address"] = v


class WorkerFailure(Exception):
    def __init__(self, worker_address: str, reason: str, details: Optional[str] = None):
        super().__init__()
        self.worker_address = worker_address
        self.reason = reason
        self.details = details


@dataclass
class Workerset:
    _workers: dict[int, Worker] = field(init=False, default_factory=dict)

    @property
    def workers(self) -> Generator[Worker, None, None]:
        for w in self._workers.values():
            yield w

    def add(self, worker: Worker, id: int):
        # if not worker.address in self._workers:
        self._workers[id] = worker

    def __getitem__(self, id: int) -> Worker:
        return self._workers[id]

    def __len__(self) -> int:
        return len(self._workers)


def start_trainer_process(
    game_config: GameConfiguration,
    env_config: EnvConfig,
    training_config: TrainingConfig,
    host: str,
    port: int,
):
    def input_(ioctx):
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                host,
                port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None

    env = builder.setup_env(game_config, env_config)
    trainer_params = TrainingParams(
        **training_config.dict(),
        observation_space=env.observation_space,
        action_space=env.action_space,
        input_=input_,
    )
    trainer = Trainer(trainer_params)
    trainer.run()


@dataclass(unsafe_hash=True)
class Worker:
    _synced: bool = field(init=False, default=False, compare=False)
    _running: bool = field(init=False, default=False, compare=False)
    _protocol_prefix: str = field(init=False, default="http://")
    address: str = field(compare=True)

    @property
    def synced(self) -> bool:
        return self._synced

    @property
    def running(self) -> bool:
        return self._running

    def sync(
        self, game_config: GameConfiguration, env_config: EnvConfig, policy_address: str
    ) -> None:
        try:
            response = requests.post(
                self._protocol_prefix + self.address + "/worker/sync",
                json={
                    "game_config": json.loads(game_config.json()),
                    "env_config": json.loads(env_config.json()),
                    "policy_address": policy_address.split(
                        ":"
                    ),  # TODO: make policy address more explicit than str
                },
            )
            if response.status_code != HTTPStatus.OK.value:
                raise WorkerFailure(
                    self.address,
                    "wrong status code",
                    f"{response.content}, {response.status_code}",
                )
        except requests.ConnectionError as exc:
            raise WorkerFailure(self.address, "unable to establis connection", str(exc))
        self._synced = True

    def start(self) -> None:
        if not self.synced:
            raise WorkerFailure(self.address, "worker is not synchornized")
        if self.running:
            raise WorkerFailure(self.address, "worker is already running")

        try:
            response = requests.post(
                self._protocol_prefix + self.address + "/worker/start"
            )
            if response.status_code != HTTPStatus.OK.value:
                raise WorkerFailure(
                    self.address,
                    "wrong status code",
                    f"{response.content}, {response.status_code}",
                )
        except requests.ConnectionError as exc:
            raise WorkerFailure(self.address, "unable to establis connection", str(exc))
        self._running = True

    def stop(self) -> None:
        if not self.synced:
            raise WorkerFailure(self.address, "worker is not synchornized")
        if not self.running:
            raise WorkerFailure(self.address, "worker is not running")

        try:
            response = requests.post(
                self._protocol_prefix + self.address + "/worker/stop"
            )
            if response.status_code != HTTPStatus.OK.value:
                raise WorkerFailure(
                    self.address,
                    "wrong status code",
                    f"{response.content}, {response.status_code}",
                )
        except requests.ConnectionError as exc:
            raise WorkerFailure(
                self.address, "unable to establish connection", str(exc)
            )
        self._running = False
