from pydantic import BaseModel, PositiveInt, ConstrainedFloat

_fraction = ConstrainedFloat(strict=True, ge=0, le=1)


class ObservationConfig(BaseModel):
    ROI: tuple[_fraction, _fraction, _fraction, _fraction] # Region of Interest relative to original observation - w0, w1, h0, h1 
    stack_size: PositiveInt
