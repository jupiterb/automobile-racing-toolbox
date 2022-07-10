from pydantic import BaseModel


class ObservationConfiguration(BaseModel):
    offset: int
    buffer_size: int
    