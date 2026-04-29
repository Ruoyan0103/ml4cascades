from .plots import TurbogapCascadePlotter, LammpsCascadePlotter
from .param import ParameterGetter
from .visualize import LammpsCascadeVisualizer
__all__ = ["TurbogapCascadePlotter", "LammpsCascadePlotter", "ParameterGetter", "LammpsCascadeVisualizer"]

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

