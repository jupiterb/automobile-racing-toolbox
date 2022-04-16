from typing import Generic, TypeVar, Any, Callable
from repository.abstract_repository import RepositoryId, RepositoryItem, AbstractRepository
from utils.custom_exceptions import ItemNotFound


RepositoryParentId = TypeVar("RepositoryParentId")


class GuardRepository(AbstractRepository, Generic[RepositoryParentId, RepositoryId, RepositoryItem]):

    def __init__(self, 
            child_repository: AbstractRepository[RepositoryId, RepositoryItem], 
            parent_repository: AbstractRepository[RepositoryParentId, Any],
            get_parnet_id: Callable[[RepositoryId], RepositoryParentId]
        ) -> None:
        self._child_repository = child_repository
        self._parent_repository = parent_repository
        self._get_parnet_id = get_parnet_id

    def get_all(self, predicate: Callable[[RepositoryId], bool] = lambda id: True) -> list[RepositoryItem]:
        return self._child_repository.get_all(lambda id: predicate(id) and self._access(id))

    def get_item(self, id: RepositoryId) -> RepositoryItem:
        if self._access(id):
            return self._child_repository.get_item(id)

    def add_item(self, id: RepositoryId, item: RepositoryItem) -> tuple[bool, RepositoryItem]:
        if self._access(id):
            return self._child_repository.add_item(id, item)

    def delete_item(self, id: RepositoryId):
        self._child_repository.delete_item(id)

    def delete_when(self, predicate: Callable[[RepositoryId], bool]):
        self._child_repository.delete_when(predicate)

    def update_item(self, id: RepositoryId, **kwargs) -> RepositoryItem:
        if self._access(id):
            self._child_repository.update_item(id, kwargs)

    def __contains__(self, id: RepositoryId) -> bool:
        return id in self._child_repository

    def _access(self, id: RepositoryParentId) -> bool:
        parent_id = self._get_parnet_id(id)
        if parent_id in self._parent_repository:
            return True
        else:
            raise ItemNotFound(str(parent_id))
