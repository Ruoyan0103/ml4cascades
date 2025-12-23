from .plots import CascadePloter

__all__ = ["CascadePloter"]

class BasicCellInfo:
    def __init__(
        self, 
        element: list[str], 
        mass: float, 
        lattice: str, 
        alat: list[float]
    ):
        self.element = element
        self.mass = mass
        self.lattice = lattice
        self.alat = alat 

