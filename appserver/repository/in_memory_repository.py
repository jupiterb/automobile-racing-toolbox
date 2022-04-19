from typing import Generic, Callable

from repository.common import GenericRepository, RepositoryId, RepositoryItem
from utils.custom_exceptions import ItemNotFound


class InMemoryRepository(GenericRepository, Generic[RepositoryId, RepositoryItem]):

    def __init__(self) -> None:
        self._data: dict[RepositoryId, RepositoryItem] = {}

    def get_all(self, predicate: Callable[[RepositoryId], bool] = lambda id: True) -> list[RepositoryItem]:
        return [value for id, value in self._data.items() if predicate(id)]

    def get_item(self, id: RepositoryId) -> RepositoryItem:
        if id in self:
            return self._data[id]
        else:
            raise ItemNotFound(item_name=str(id))

    def add_item(self, id: RepositoryId, item: RepositoryItem) -> tuple[bool, RepositoryItem]:
        if id in self:
            return (False, self._data[id])
        else:
            self._data[id] = item
            return (True, item)

    def delete_item(self, id: RepositoryId):
        if id in self:
            del self._data[id]

    def delete_when(self, predicate: Callable[[RepositoryId], bool]):
        ids = [id for id in self._data.keys() if predicate(id)] 
        for id in ids:
            del self._data[id]

    def update_item(self, id: RepositoryId, **kwargs) -> RepositoryItem:
        item = self.get_item(id)
        for key, value in kwargs.items():
            if key in item.__dict__: 
                item.__dict__[key] = value
        return item

    def __contains__(self, id: RepositoryId) -> bool:
        return id in self._data
