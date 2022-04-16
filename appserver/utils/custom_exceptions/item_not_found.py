class ItemNotFound(Exception):

    def __init__(self, item_name: str, *args: object) -> None:
        super().__init__(*args)
        self.item_name: str = item_name
