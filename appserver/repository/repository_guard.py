from typing import Generic, TypeVar, Any
from repository.abstract_repository import RepositoryId, RepositoryItem, AbstractRepository
from utils.custom_exceptions import ItemNotFound


RepositoryParentId = TypeVar("RepositoryParentId")


class RepositoryGurard(Generic[RepositoryParentId, RepositoryId, RepositoryItem]):

    def __init__(self, 
            owned_repository: AbstractRepository[RepositoryId, RepositoryItem], 
            guard_repository: AbstractRepository[RepositoryParentId, Any]
        ) -> None:
        super().__init__()
        self._owned_repository = owned_repository
        self._guard_repository = guard_repository

    def access(self, parent_id: RepositoryParentId) -> AbstractRepository[RepositoryId, RepositoryItem]:
        if self._guard_repository.contains(parent_id):
            return self._owned_repository
        else:
            raise ItemNotFound(str(parent_id))
