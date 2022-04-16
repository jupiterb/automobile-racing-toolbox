from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar, Generic, Callable


RepositoryId = TypeVar("RepositoryId")
RepositoryItem = TypeVar("RepositoryItem")


class AbstractRepository(Generic[RepositoryId, RepositoryItem], ABC):

    @abstractmethod
    def get_all(self, predicate: Callable[[RepositoryId], bool] = lambda id: True) -> List[RepositoryItem]:
        pass

    @abstractmethod
    def get_item(self, id: RepositoryId) -> RepositoryItem:
        pass

    @abstractmethod
    def add_item(self, id: RepositoryId, item: RepositoryItem) -> Tuple[bool, RepositoryItem]:
        pass

    @abstractmethod
    def delete_item(self, id: RepositoryId):
        pass

    @abstractmethod
    def delete_when(self, predicate: Callable[[RepositoryId], bool]):
        pass

    @abstractmethod 
    def update_item(self, id: RepositoryId, **kwargs) -> RepositoryItem:
        pass

    @abstractmethod
    def contains(self, id: RepositoryId) -> bool:
        pass
