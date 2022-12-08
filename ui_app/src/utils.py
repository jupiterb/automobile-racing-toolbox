import json


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)


class UIAppError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
