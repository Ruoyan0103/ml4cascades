from abc import ABC, abstractmethod
import os
from ml4cascades.loggers.logger import AppLogger
from .utils import TCeKappa
from .calcs_eph import CascadeCalculatorEPH

module_dir = os.path.dirname(__file__)
result_dir = os.path.join(module_dir, 'results')
log_dir = os.path.join(module_dir, 'logs')

class TurboGAPCalculator(ABC):
    def __init__(self, task_name: str):
        self.template_dir = os.path.join(module_dir, 'templates', task_name)
        self.calculation_dir = os.path.join(result_dir, task_name, potential.name)
        self.log_file = os.path.join(log_dir, f'{task_name}_{potential.name}.log')
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()

        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.calculation_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        

__all__['TCeKappa', 'CascadeCalculatorEPH']