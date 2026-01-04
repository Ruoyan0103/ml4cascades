from abc import ABC
import os
from ml4cascades.loggers.logger import AppLogger

module_dir = os.path.dirname(__file__)

class LMPSCalculator(ABC):
    def __init__(self, task_name: str, model_name: str):
        self.template_dir = os.path.join(module_dir, 'templates', task_name)
        self.result_dir = os.path.join(module_dir, 'results', task_name)
        self.log_dir = os.path.join(module_dir, 'logs', task_name)

        self.calculation_dir = os.path.join(self.result_dir, model_name)
        self.log_file = os.path.join(self.log_dir, f'{model_name}.log')
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()

        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.calculation_dir, exist_ok=True)