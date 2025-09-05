class BasicInput:
    def __init__(
        self, 
        potential: str, 
        mass: list[float], 
        element: list[str], 
        lattice: list[str], 
        alat: list[list[float]]
    ):
        self.potential = potential
        self.num_species = num_species
        self.mass = mass
        self.element = element
        self.lattice = lattice
        self.alat = alat