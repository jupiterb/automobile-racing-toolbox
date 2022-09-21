from pydantic import BaseModel
from h5py import Dataset


class Recording(BaseModel):
    game: str
    user: str
    name: str
    fps: int
    images: Dataset
    actions: Dataset
    features: Dataset

    class Config:
        arbitrary_types_allowed = True
