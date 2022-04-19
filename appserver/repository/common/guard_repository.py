from typing import Generic, TypeVar, Any, Callable
from repository.common.generic_repository import RepositoryId, RepositoryItem, GenericRepository
from utils.custom_exceptions import ItemNotFound


RepositoryParentId = TypeVar("RepositoryParentId")


class GuardRepository(GenericRepository, Generic[RepositoryParentId, RepositoryId, RepositoryItem]):

    def __init__(self, 
            child_repository: GenericRepository[RepositoryId, RepositoryItem], 
            parent_repository: GenericRepository[RepositoryParentId, Any],
            get_parnet_id: Callable[[RepositoryId], RepositoryParentId]
        ) -> None:
        self._child_repository = child_repository
        self._parent_repository = parent_repository
        self._get_parnet_id = get_parnet_id

    def get_all(self, predicate: Callable[[RepositoryId], bool] = lambda id: True) -> list[RepositoryItem]:
        return self._child_repository.get_all(lambda id: predicate(id))

    def get_item(self, id: RepositoryId) -> RepositoryItem:
        return self._with_checking_access(id).get_item(id)

    def add_item(self, id: RepositoryId, item: RepositoryItem) -> tuple[bool, RepositoryItem]:
        return self._with_checking_access(id).add_item(id, item)

    def delete_item(self, id: RepositoryId):
        try:
            self._with_checking_access(id).delete_item(id)
        except (ItemNotFound):
            pass

    def delete_when(self, predicate: Callable[[RepositoryId], bool]):
        self._child_repository.delete_when(predicate)

    def update_item(self, id: RepositoryId, **kwargs) -> RepositoryItem:
        return self._with_checking_access(id).update_item(id, **kwargs)

    def __contains__(self, id: RepositoryId) -> bool:
        return id in self._child_repository

    def _with_checking_access(self, id: RepositoryId) -> GenericRepository:
        parent_id = self._get_parnet_id(id)
        if parent_id in self._parent_repository:
            return self._child_repository
        else:
            raise ItemNotFound(str(parent_id))
