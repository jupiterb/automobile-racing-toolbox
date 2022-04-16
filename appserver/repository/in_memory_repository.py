from abc import ABC
from typing import List, Tuple, Dict, Generic, Callable

from repository.abstract_repository import AbstractRepository, RepositoryId, RepositoryItem
from utils.custom_exceptions import ItemNotFound


class InMemoryRepository(AbstractRepository, Generic[RepositoryId, RepositoryItem], ABC):

    def __init__(self) -> None:
        super().__init__()
        self._data: Dict[RepositoryId, RepositoryItem] = {}

    def get_all(self, predicate: Callable[[RepositoryId], bool] = lambda id: True) -> List[RepositoryItem]:
        return [value for id, value in self._data.items() if predicate(id)]

    def get_item(self, id: RepositoryId) -> RepositoryItem:
        if self.contains(id):
            return self._data[id]
        else:
            raise ItemNotFound(item_name=str(id))

    def add_item(self, id: RepositoryId, item: RepositoryItem) -> Tuple[bool, RepositoryItem]:
        if self.contains(id):
            return (False, self._data[id])
        else:
            self._data[id] = item
            return (True, item)

    def delete_item(self, id: RepositoryId):
        if self.contains(id):
            del self._data[id]

    def delete_when(self, predicate: Callable[[RepositoryId], bool]):
        ids = [id for id in self._data.keys() if predicate(id)] 
        for id in ids:
            del self._data[id]

    def update_item(self, id: RepositoryId, **kwargs) -> RepositoryItem:
        item = self.get_item(id)
        for key, value in kwargs.items():
            item.__dict__[key] = value
        return item

    def contains(self, id: RepositoryId) -> bool:
        return id in self._data
