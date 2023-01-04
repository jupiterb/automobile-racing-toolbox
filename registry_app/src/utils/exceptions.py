class RegistryAppException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    @property
    def message(self):
        return self._message


class UserExistsException(RegistryAppException):
    def __init__(self, message: str) -> None:
        super().__init__(message)
