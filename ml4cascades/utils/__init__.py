from .plots import TurbogapCascadePloter, LammpsCascadePloter
from .param import ParameterGetter
__all__ = ["TurbogapCascadePloter", "LammpsCascadePloter", "ParameterGetter"]

class BasicCellInfo:
    def __init__(
        self, 
        element: list[str], 
        atomic_num: list[int],
        mass: float, 
        lattice: str, 
        alat: list[float]
    ):
        self.element = element
        self.atomic_num = atomic_num
        self.mass = mass
        self.lattice = lattice
        self.alat = alat 

