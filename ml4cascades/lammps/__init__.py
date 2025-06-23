from abc import ABC, abstractmethod
class LMPStaticCalculator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _setup(self):
        """
        Setup the input file for the LAMMPS simulation.
        """
        pass

    @abstractmethod
    def calculate(self):
        """
        Calculate the properties using LAMMPS.
        """
        pass