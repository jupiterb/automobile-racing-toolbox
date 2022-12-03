from pydantic import BaseModel, Field
from abc import abstractmethod
from threading import Lock
import uuid


class RemoteWorkerRef(BaseModel):
    address: str
    game_id: str
    id_: uuid.UUID = Field(default_factory=uuid.uuid1)

    class Config:
        frozen = True


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class RemoteWorkerRegistry(metaclass=SingletonMeta):
    @abstractmethod
    def register_worker(self, worker_ref: RemoteWorkerRef) -> None:
        """Cal this method to add new worker to registry. Raise adequate exception, if worker already exist"""

    @abstractmethod
    def remove_worker(self, worker_id: uuid.UUID) -> None:
        """Call this method to remove worker for registry. Raise exception if it doesn't exist"""

    @abstractmethod
    def update_timestamp(self, worker_id: uuid.UUID) -> None:
        """Call this method to mark that worker is still alive"""

    @abstractmethod
    def get_workers(self, game_id: str) -> set[RemoteWorkerRef]:
        """Returns active workers that can run given game"""

    @abstractmethod
    def get_active_workers(self) -> set[RemoteWorkerRef]:
        ...
