class WindowNotFound(Exception):

    def __init__(self, process_name: str, *args: object) -> None:
        super().__init__(*args)
        self.process_name: str = process_name
