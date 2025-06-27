from abc import ABC, abstractmethod
import os
from ml4cascades.loggers.logger import AppLogger

module_dir = os.path.dirname(__file__)
result_dir = os.path.join(module_dir, 'results')
log_dir = os.path.join(module_dir, 'logs')


class TurboGAPCalculator(ABC):
    def __init__(self, task_name, potential, num_species, mass, element, lattice, alat):
        """
        Initialize the LAMMPS static calculator.
        Args:
            task_name (str): Name of the task.
            num_species (int): Number of species in the system. 
            potential (Potential): The potential object containing force field settings.
            num_species (int): Number of species in the system.
            mass (float): Mass of the atoms.
            element (str): Element symbol.
            lattice (str): Lattice type.
            alat (float): Lattice constant.
        """
        self.template_dir = os.path.join(module_dir, 'templates', task_name)
        self.calculation_dir = os.path.join(result_dir, task_name, potential.name)
        self.log_file = os.path.join(log_dir, f'{task_name}_{potential.name}.log')

        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.calculation_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.task_name = task_name
        self.num_species = num_species
        self.ff_settings = potential.ff_settings
        self.mass = mass
        self.element = element
        self.lattice = lattice
        self.alat = alat
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()
        

    @abstractmethod
    def _setup(self):
        pass
        

    @abstractmethod
    def calculate(self):
        """
        Calculate the properties using LAMMPS.
        """
        pass