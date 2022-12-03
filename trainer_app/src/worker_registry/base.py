from dataclasses import dataclass, field
from abc import abstractmethod
import uuid


@dataclass(frozen=True)
class RemoteWorkerRef:
    address: str
    game_id: str
    id_: uuid.UUID = field(init=False, default_factory=uuid.uuid1)


class SingletonType(type):
    def __new__(mcls, name, bases, attrs):
        # Assume the target class is created (i.e. this method to be called) in the main thread.
        cls = super(SingletonType, mcls).__new__(mcls, name, bases, attrs)
        cls.__shared_instance_lock__ = Lock()
        return cls

    def __call__(cls, *args, **kwargs):
        with cls.__shared_instance_lock__:
            try:
                return cls.__shared_instance__
            except AttributeError:
                cls.__shared_instance__ = super(SingletonType, cls).__call__(
                    *args, **kwargs
                )
                return cls.__shared_instance__


class RemoteWorkerRegistry(metaclass=SingletonType):
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
