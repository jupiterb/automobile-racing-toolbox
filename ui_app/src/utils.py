import json
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)


class TaskInfo(BaseModel):
    task_finish_time: Optional[datetime]
    task_name: Optional[str]
    task_id: Optional[str]
    status: Optional[str]
    result: Optional[Any]
