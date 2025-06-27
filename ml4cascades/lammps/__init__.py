from abc import ABC, abstractmethod
import os
from ml4cascades.loggers.logger import AppLogger

module_dir = os.path.dirname(__file__)
result_dir = os.path.join(module_dir, 'results')
log_dir = os.path.join(module_dir, 'logs')


class LMPStaticCalculator(ABC):
    def __init__(self, task_name, potential, mass, element, lattice, alat):
        """
        Initialize the LAMMPS static calculator.
        Args:
            task_name (str): Name of the task.
            potential (Potential): The potential object containing force field settings.
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
        self.ff_settings = potential.ff_settings
        self.mass = mass
        self.element = element
        self.lattice = lattice
        self.alat = alat
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()
        

    @abstractmethod
    def _setup(self, exe_folder=None):
        """
        Setup the input file for the LAMMPS simulation.
        Args:
            exe_folder (str, optional): Directory where the executable files will be placed.
        """
        with open(os.path.join(self.template_dir, 'submit.sh'), 'r') as f:
            submit_template = f.read()
        if exe_folder is None:
            exe_folder = self.calculation_dir
        submit_file = os.path.join(exe_folder, 'submit.sh')
        with open(submit_file, 'w') as f:
            f.write(submit_template.format(file=f'in.{self.task_name}'))


    @abstractmethod
    def calculate(self):
        """
        Calculate the properties using LAMMPS.
        """
        pass