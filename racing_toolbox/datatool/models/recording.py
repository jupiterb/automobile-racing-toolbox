from pydantic import BaseModel
from h5py import Dataset
from numpy import ndarray


class Recording(BaseModel):
    game: str
    user: str
    name: str
    fps: int
    images: Dataset | ndarray
    actions: Dataset | ndarray
    features: Dataset | ndarray

    class Config:
        arbitrary_types_allowed = True
