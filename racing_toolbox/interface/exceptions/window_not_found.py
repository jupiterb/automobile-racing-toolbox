class WindowNotFound(Exception):
    def __init__(self, process_name: str) -> None:
        super().__init__(f"Process {process_name} not found")
