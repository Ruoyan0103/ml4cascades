from ml4cascades.potentials import IPotential

class BasicInput:
    def __init__(
        self, 
        potential: IPotential, 
        mass: float, 
        element: str, 
        lattice: str, 
        alat: list[float]
    ):
        self.potential = potential
        self.mass = mass
        self.element = element
        self.lattice = lattice
        self.alat = alat