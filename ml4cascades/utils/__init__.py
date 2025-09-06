from ml4cascades.potentials import IPotential
class BasicInput:
    def __init__(
        self, 
        potential: IPotential, 
        mass: list[float], 
        element: list[str], 
        lattice: list[str], 
        alat: list[list[float]]
    ):
        self.ff_settings = ff_settings
        self.mass = mass
        self.element = element
        self.lattice = lattice
        self.alat = alat