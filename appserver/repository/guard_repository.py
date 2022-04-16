from abc import ABC
from typing import Generic, TypeVar, Any, Callable, List, Tuple
from repository.abstract_repository import RepositoryId, RepositoryItem, AbstractRepository
from utils.custom_exceptions import ItemNotFound


RepositoryParentId = TypeVar("RepositoryParentId")


class GuardRepository(AbstractRepository, Generic[RepositoryParentId, RepositoryId, RepositoryItem]):

    def __init__(self, 
            owned_repository: AbstractRepository[RepositoryId, RepositoryItem], 
            guard_repository: AbstractRepository[RepositoryParentId, Any],
            get_parnet_id: Callable[[RepositoryId], RepositoryParentId]
        ) -> None:
        super().__init__()
        self._owned_repository = owned_repository
        self._guard_repository = guard_repository
        self._get_parnet_id = get_parnet_id

    def get_all(self, predicate: Callable[[RepositoryId], bool] = lambda id: True) -> List[RepositoryItem]:
        return self._owned_repository.get_all(lambda id: predicate(id) and self._access(id))

    def get_item(self, id: RepositoryId) -> RepositoryItem:
        if self._access(id):
            return self._owned_repository.get_item(id)

    def add_item(self, id: RepositoryId, item: RepositoryItem) -> Tuple[bool, RepositoryItem]:
        if self._access(id):
            return self._owned_repository.add_item(id, item)

    def delete_item(self, id: RepositoryId):
        self._owned_repository.delete_item(id)

    def delete_when(self, predicate: Callable[[RepositoryId], bool]):
        self._owned_repository.delete_when(predicate)

    def update_item(self, id: RepositoryId, **kwargs) -> RepositoryItem:
        if self._access(id):
            self._owned_repository.update_item(id, kwargs)

    def contains(self, id: RepositoryId) -> bool:
        return self._owned_repository.contains(id)

    def _access(self, id: RepositoryParentId) -> bool:
        parent_id = self._get_parnet_id(id)
        if self._guard_repository.contains(parent_id):
            return True
        else:
            raise ItemNotFound(str(parent_id))
