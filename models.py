from dataclasses import dataclass

@dataclass(slots=True)
class Track:
    box: object
    name: str
    conf: float
    id: int
    missed: int = 0
    name_id: int = 0

@dataclass(slots=True)
class Detection:
    box: object
    name: str
    conf: float
    name_id: int = 0