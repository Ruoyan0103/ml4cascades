from ml4cascades.lammps import LMPStaticCalculator
import os 

module_dir = os.path.dirname(__file__)

class RelaxationCalculator(LMPStaticCalculator):
    def __init__(self):
        super().__init__()

    def _setup(self):
        """
        Setup the input file for the LAMMPS relaxation simulation.
        """
        self.lmp.command("min_style cg")
        self.lmp.command("min_modify dmax 0.01")

    def calculate(self):
        """
        Perform the relaxation calculation using LAMMPS.
        """
        self.lmp.command("minimize 1e-6 1e-8 1000 10000")


class CascadeSimulationCalculator(LMPStaticCalculator):
    def __init__(self):
        super().__init__()

    def _setup(self):
        """
        Setup the input file for the LAMMPS relaxation simulation.
        """
        self.lmp.command("min_style cg")
        self.lmp.command("min_modify dmax 0.01")

    def calculate(self):
        """
        Perform the relaxation calculation using LAMMPS.
        """
        self.lmp.command("minimize 1e-6 1e-8 1000 10000")