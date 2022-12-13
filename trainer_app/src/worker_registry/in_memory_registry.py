from src.worker_registry.base import (
    RemoteWorkerRef,
    RemoteWorkerRegistry,
    SingletonMeta,
)
from src.worker_registry.exceptions import RecordExists, RecordDoesntExist
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Lock
from threading import get_ident
import uuid
from logging import getLogger
from datetime import datetime, timedelta

logger = getLogger(__name__)


def synchornized(lock):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            # with lock:
            result = fun(*args, **kwargs)
            return result

        return wrapper

    return decorator


class MemoryRegistry(RemoteWorkerRegistry):
    _lock = Lock()

    def __init__(self, expiry_time: timedelta):
        self._expiry_time = expiry_time
        self._workers: list[RemoteWorkerRef] = list()
        self._id2worker: dict[uuid.UUID, RemoteWorkerRef] = {}
        self._id2timestamp: dict[uuid.UUID, datetime] = {}

    @synchornized(_lock)
    def register_worker(self, worker_ref: RemoteWorkerRef) -> None:
        if worker_ref in self._workers:
            raise RecordExists(worker_ref)
        self._workers.append(worker_ref)
        self._id2worker[worker_ref.id_] = worker_ref
        self._id2timestamp[worker_ref.id_] = datetime.now()
        logger.debug(f"worker added: {worker_ref}")

    @synchornized(_lock)
    def remove_worker(self, worker_id: uuid.UUID) -> None:
        """Call this method to remove worker for registryt"""
        if worker_id not in self._id2worker:
            logger.warning(f"Trying to remove non existing worker: {worker_id}")
            raise RecordDoesntExist(worker_id)
        self._workers = [w for w in self._workers if w.id_ != worker_id]
        del self._id2worker[worker_id]
        del self._id2timestamp[worker_id]
        logger.debug(f"Removed worker: {worker_id}")

    @synchornized(_lock)
    def update_timestamp(self, worker_id: uuid.UUID, available: bool) -> None:
        """Call this method to mark that worker is still alive"""
        if worker_id not in self._id2timestamp:
            logger.warning(
                f"Trying to update timestamp for non existing worker: {worker_id}"
            )
            raise RecordDoesntExist(worker_id)
        self._id2timestamp[worker_id] = datetime.now()
        self._id2worker[worker_id].available = available

    @synchornized(_lock)
    def get_workers(self, game_id: str) -> list[RemoteWorkerRef]:
        """Returns active workers that can run given game"""
        active_workers = self.get_active_workers()
        return [w for w in active_workers if w.game_id == game_id]

    @synchornized(_lock)
    def get_active_workers(self) -> list[RemoteWorkerRef]:
        now = datetime.now()
        return [
            w
            for w in self._workers
            if now - self._id2timestamp[w.id_] < self._expiry_time
        ]


def get_registry():
    return MemoryRegistry(timedelta(seconds=15))
