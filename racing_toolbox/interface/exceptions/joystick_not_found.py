class JoystickNotFound(Exception):
    def __init__(self) -> None:
        super().__init__(f"Any foystck not found")
