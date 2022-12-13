class ItemExists(Exception):
    def __init__(
        self, game_name: str, user_name: str, item_name: str, *args: object
    ) -> None:
        super().__init__(*args)
        self.game_name: str = game_name
        self.user_name: str = user_name
        self.item_name: str = item_name
