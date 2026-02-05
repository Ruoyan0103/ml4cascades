import os, shutil
import numpy as np
from ml4cascades.loggers.logger import AppLogger

module_dir = os.path.dirname(__file__)

class CascadeChecker:
    def __init__(self,
                 PKA_kin_eng: float,
                 supercell_size: list[int],
                 radius_frac: float,
                 traj_folder: str,
                 grid: int, 
                 task_name='checking',
                 model_name='EPH'):
        self.PKA_kin_eng = PKA_kin_eng
        self.supercell_size = supercell_size
        self.radius_frac = radius_frac
        self.traj_folder = traj_folder
        self.grid = grid
        self.log_dir = os.path.join(module_dir, 'logs', task_name)
        self.log_file = os.path.join(self.log_dir, f'{model_name}.log')
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()
        os.makedirs(self.log_dir, exist_ok=True)
        
    def check_output(self, 
                     start_output: int,
                     num_outputs: int, 
                     running_time: float):
        self.logger.info(f'#---------Checking, PKA energy: {self.PKA_kin_eng}, supercellsize: {self.supercell_size[0]}-{self.supercell_size[1]}-{self.supercell_size[2]}, radius_frac: {self.radius_frac}, grid: {self.grid}---------#')
        failed_case = 0
        for num_output in range(num_outputs):
            output_file = os.path.join(self.traj_folder, f'{start_output+num_output}', 'thermo.out')
            if not os.path.exists(output_file):
                self.logger.info(f'Output {start_output+num_output} not found.')
                continue
            data = np.loadtxt(output_file, skiprows=1)
            time_end = data[-1, 1]
            if time_end < running_time:
                failed_case += 1
                self.logger.info(f'Output {start_output+num_output} failed, running time: {time_end} ps.')
            # else:
            #     src_folder = os.path.join(self.traj_folder, f'{start_output+num_output}')
            #     dst_folder = os.path.join(self.successful_folder, f'{start_output+num_output}')
            #     shutil.move(src_folder, dst_folder)
            #     self.logger.info(f'Moving from {src_folder} to {dst_folder}.')
        self.logger.info(f'#---------{failed_case}/{num_outputs} cases failed.---------#')
